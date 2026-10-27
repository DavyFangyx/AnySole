"""Numerical A2 contracts; synthetic fixtures are not claimed as data evidence."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anysole import train
from anysole.models import visual
from anysole.models.common.pressure_a2 import QueryPressureHead, CapacityPressureHead, split_feet, join_feet
from anysole.utils.a2_config import a2_budget, validate_a2_config, A2_RUNS, A3_RUNS, A3_VERSION
from anysole.utils.a2_gradients import parameter_groups, routed_backward, gradient_statistics
from anysole.utils.a2_training import initialization_identity, fixed_gradient_probe
from anysole.utils.a2_pressure_losses import a2_pressure_counts, modify_pressure_terms, calibrate_auxiliary_coefficient
from anysole.utils.losses import compute_visual_loss_terms, compute_pair_loss_terms
from test_view_losses import batch_fixture, view_index, predictions, config as loss_config
from test_view_training import ToyGroups, config as training_config, validation, assert_state_equal
from anysole.data.dataset import grouped_collate


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def cfg(run="G-HF", **extra):
    experiment, head, pose, pressure, pair, supervised = A2_RUNS[run]
    return {"a2_version": "anysole.a2.v1", "a2_run_id": run, "seed": 1,
            "pressure_pair_route": pair, "supervised_pressure_route": supervised,
            "budget_mode": "cam3_exact" if run == "B-CAM3" else "standard",
            "experiment": experiment, "pressure_head": head, "pressure_time_pe": "before_fusion",
            "lambda_mv": .2 if pose else 0., "mv_pose_weight": 1.,
            "mv_pressure_weight": pressure/.2 if pose else 0.,
            "effective_pose_pair_peak": pose, "effective_pressure_pair_peak": pressure,
            "a2_head_seed": 1, "capacity_width": 544, "dropout": .1, **extra}


class TinyRoute(nn.Module):
    def __init__(self, dropout=.2):
        super().__init__()
        self.embeddings = nn.Linear(3, 8, bias=False)
        self.encoders = nn.Module()
        self.encoders.v_enc = nn.Module()
        self.encoders.v_enc.proj = nn.Linear(3, 8)
        self.encoders.v_enc.norm = nn.LayerNorm(8)
        self.encoders.v_enc.shared_alias = self.embeddings
        self.fusion = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Dropout(dropout))
        self.aux_heads = nn.Module()
        self.aux_heads.t_rec = nn.Linear(8, 96)
        self.pose = nn.Linear(8, 144)
        self.velocity = nn.Linear(8, 3)
        self.forward_calls = 0

    def forward_visual(self, features, valid, timestamps):
        self.forward_calls += 1
        x = features[..., :3]
        h = self.fusion(self.encoders.v_enc.norm(self.encoders.v_enc.proj(x)) + self.embeddings(x))
        identity = h.new_tensor([1, 0, 0, 0, 1, 0]).repeat(24)
        return {"pressure_hat": self.aux_heads.t_rec(h), "x0_hat": identity+.05*self.pose(h),
                "v_hat": self.velocity(h), "sigma_frozen": False}


def test_schema_and_budget_are_explicit_and_do_not_change_old_identity():
    assert a2_budget()["target_updates"] == 5180
    b = a2_budget("cam3_exact")
    assert b["target_updates"] == 20720 and b["target_rgb_windows"] == 5304320
    assert b["lr_warmup_updates"] == 1036 and b["sigma_freeze_updates"] == 2072
    assert len(list(range(400, 20720, 400)))+1 == 52
    validate_a2_config(cfg())
    for damaged in ({"pressure_pair_route": "all"}, {"effective_pressure_pair_peak": .4},
                    {"mv_pressure_weight": 2}, {"a2_version": "wrong"}, {"seed": 2}):
        with pytest.raises(ValueError):
            validate_a2_config({**cfg(), **damaged})
    with pytest.raises(ValueError, match="explicit a2_version"):
        validate_a2_config({"pressure_pair_route": "HF"})
    old = visual.effective_config({"pressure_head": "time_pe"})
    same = visual.effective_config({**cfg(), "pressure_pair_route": "H"})
    assert old["structure_hash"] == same["structure_hash"]
    assert "a2_version" not in old and "pressure_pair_route" not in old
    assert train.visual_run_identity({**old, "a2_version": "anysole.a2.v1"}) != train.visual_run_identity(old)


@pytest.mark.parametrize("route", ["H", "HF"])
def test_isolated_route_is_parameter_identity_whitelist_and_both_pair_ends_live(route):
    torch.manual_seed(51)
    m = TinyRoute(dropout=0)
    x = torch.randn(2, 5, 2051)
    out = m.forward_visual(x, torch.ones(2, 5, dtype=torch.bool), torch.arange(5)[None].expand(2,-1).double())
    out["pressure_hat"].retain_grad()
    value = (out["pressure_hat"][0]-out["pressure_hat"][1]).square().mean()
    routed_backward({"pressure_pair": value}, m, {"pressure_pair_route": route}, torch.amp.GradScaler("cuda", enabled=False))
    groups = parameter_groups(m)
    assert not {id(p) for p in groups["E_shared"]} & {id(p) for p in groups["R"]}
    allowed = {"H", "F"} if route == "HF" else {"H"}
    for name, parameters in groups.items():
        norm = sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)
        assert norm > 0 if name in allowed else norm == 0
    # The original pair loss is symmetric; autograd restricts parameter inputs,
    # never a prediction endpoint. Check explicit unconstrained endpoint grads.
    a = torch.autograd.grad((out["pressure_hat"][0]-out["pressure_hat"][1]).square().mean(),
                            out["pressure_hat"], retain_graph=True)[0]
    assert a[0].abs().sum() > 0 and a[1].abs().sum() > 0


@pytest.mark.parametrize("restricted", ["pressure_pair", "pressure"])
def test_joint_gradient_is_sum_of_per_loss_masks_other_rgb_gradients_survive(restricted):
    torch.manual_seed(13)
    m = TinyRoute(dropout=0)
    batch = batch_fixture((2,), tw=5, dtype=torch.float32)
    out = train.visual_forward(m, batch, view_index(batch))
    sup = compute_visual_loss_terms(out, batch, view_index(batch), loss_config())
    pair = compute_pair_loss_terms(out, batch, view_index(batch), loss_config())
    terms = {"pose": .3*sup["pose"]["raw_sum"]/sup["pose"]["valid_count"],
             "pressure": .1*sup["pressure"]["raw_sum"]/sup["pressure"]["valid_count"],
             "pressure_pair": .2*pair["pressure_pair"]["raw_sum"]/pair["pressure_pair"]["valid_count"]}
    parameters = list(m.parameters())
    route_cfg = {"pressure_pair_route": "HF" if restricted == "pressure_pair" else "all",
                 "supervised_pressure_route": "HF" if restricted == "pressure" else "all"}
    allowed = {id(p) for n in ("H", "F") for p in parameter_groups(m)[n]}
    expected = [torch.zeros_like(p) for p in parameters]
    for name, value in terms.items():
        grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        for i, (p, grad) in enumerate(zip(parameters, grads)):
            if grad is not None and (name != restricted or id(p) in allowed):
                expected[i] += grad
    routed_backward(terms, m, route_cfg, torch.amp.GradScaler("cuda", enabled=False))
    for p, value in zip(parameters, expected):
        torch.testing.assert_close(p.grad if p.grad is not None else torch.zeros_like(p), value, rtol=2e-6, atol=1e-8)
    assert sum(float(p.grad.abs().sum()) for p in parameter_groups(m)["R"] if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in parameter_groups(m)["E_shared"] if p.grad is not None) > 0
    assert m.forward_calls == 1


@pytest.mark.parametrize("kind", ["query_direct", "query_nonnegative", "query_factorized"])
def test_query_shape_safe_all_missing_mask_and_single_view_memory(kind):
    torch.manual_seed(4)
    mean = torch.linspace(.01, .4, 96)
    head = QueryPressureHead(kind, dropout=0, cell_mean=mean, contact_rate=[.3,.7]).eval()
    valid = torch.tensor([[True,False,True,True], [False]*4, [True]*4])
    memory = torch.randn(3, 8, 256)
    time = torch.arange(4).double()[None].expand(3,-1)/40
    captured = []
    hook = head.attention.register_forward_pre_hook(lambda module,args,kwargs: captured.append(kwargs["key_padding_mask"].clone()), with_kwargs=True)
    first = head(memory, valid, time)
    hook.remove()
    assert torch.equal(captured[0], ~valid[valid.any(-1)])
    altered = memory.clone()
    altered[:, 4:] = 1e5  # tactile memory is not a query key/value
    altered[0, 1] = float("nan")  # ignored visual key must be removed first
    second = head(altered, valid, time)
    torch.testing.assert_close(first["pressure_hat"], second["pressure_hat"], rtol=0, atol=0)
    assert first["pressure_hat"].shape == (3,4,96)
    assert torch.isfinite(first["pressure_hat"]).all() and (first["pressure_hat"][~valid] == 0).all()
    assert torch.equal(first["pressure_query_valid"], valid[..., None].expand(-1,-1,2))
    if kind != "query_direct":
        torch.testing.assert_close(first["pressure_hat"][valid], mean.expand(int(valid.sum()),-1), rtol=3e-6, atol=1e-7)
        assert (first["pressure_hat"] >= 0).all()
        if kind == "query_factorized":
            torch.testing.assert_close(first["pressure_allocation"][valid].sum(-1), torch.ones(int(valid.sum()),2))
            torch.testing.assert_close(split_feet(first["pressure_hat"]).sum(-1), first["pressure_total"])
    clone = copy.deepcopy(head)
    clone.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(clone(memory, valid, time)["pressure_hat"], first["pressure_hat"], rtol=0, atol=0)
    pressure = torch.arange(96.).reshape(1,1,96)
    assert torch.equal(join_feet(split_feet(pressure)), pressure)
    assert split_feet(pressure)[0,0,1,0] == 48


def test_capacity_matching_and_no_dummy_capacity():
    q = QueryPressureHead(dropout=0)
    c = CapacityPressureHead()
    counts = [sum(p.numel() for p in h.parameters()) for h in (q,c)]
    assert abs(counts[0]-counts[1])/counts[0] <= .05
    out = c(torch.randn(2,40,256), torch.ones(2,20,dtype=torch.bool), torch.arange(20)[None].expand(2,-1).double()/40)
    out["pressure_hat"].square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in c.parameters())
    nnhead = QueryPressureHead("query_nonnegative", cell_mean=[.1]*96, contact_rate=[.4,.6])
    fact = QueryPressureHead("query_factorized", cell_mean=[.1]*96, contact_rate=[.4,.6])
    a,b = (sum(p.numel() for p in h.parameters()) for h in (nnhead,fact))
    assert abs(a-b)/a <= .05


def test_new_head_rng_preserves_common_weights_and_pair_initialization():
    identities, states = {}, {}
    for run in ("W010", "C0", "CP", "Q0", "QP", "Q-NN", "Q-FACT"):
        torch.manual_seed(1)
        model = visual.build(cfg(run, pressure_cell_mean=[.1]*48+[.2]*48, pressure_contact_rate=[.4,.6]))
        identities[run] = initialization_identity(model)
        states[run] = torch.get_rng_state()
        del model
    assert len({v["common_state_hash"] for v in identities.values()}) == 1
    assert len({bytes(s.tolist()) for s in states.values()}) == 1
    assert identities["C0"]["pressure_state_hash"] == identities["CP"]["pressure_state_hash"]
    assert identities["Q0"]["pressure_state_hash"] == identities["QP"]["pressure_state_hash"]
    assert identities["Q-NN"]["query_state_hash"] == identities["Q-FACT"]["query_state_hash"]


def test_optional_loss_count_masks_and_actual_dt():
    batch = batch_fixture((2,), tw=5, dtype=torch.float32)
    batch["visual"]["view_edges"][0,0,1] = False
    batch["masks"]["pressure_edges"][0,2,:48] = False
    batch["masks"]["pressure_valid"][0,1,48:] = False
    index = view_index(batch)
    out = predictions(batch,index)
    out["pressure_hat"] = out["pressure_hat"] + .5
    for loss in ("total", "diff", "active", "huber"):
        config = loss_config(a2_version="anysole.a2.v1", pressure_loss=loss, pressure_diff_delta=[.1,.2],
                             pressure_huber_delta=[.1,.2], pressure_active_threshold=[0.]*96,
                             pressure_active_alpha=2., pressure_active_normalizer=3.)
        terms = compute_visual_loss_terms(out,batch,index,config)
        counts = a2_pressure_counts(batch,index,config)
        for name,count in counts.items():
            assert terms[name]["valid_count"] == count
        if loss == "total":
            torch.testing.assert_close(terms["pressure_total"]["raw_sum"]/counts["pressure_total"], torch.tensor(.25))
        if loss == "diff":
            assert terms["pressure_diff"]["raw_sum"] < 1e-12
        assert torch.isfinite(terms["pressure"]["raw_sum"])


def test_gradient_probe_keeps_rng_mode_and_gradients_and_zero_cosine_is_na():
    torch.manual_seed(4)
    m = TinyRoute()
    batch = grouped_collate([ToyGroups()[0], ToyGroups()[1]])
    before = train.capture_training_rng(torch.device("cpu"))
    state = copy.deepcopy(m.state_dict())
    result = fixed_gradient_probe(m,batch,{**loss_config(), "tw":20,"lambda_mv":.2},torch.device("cpu"),update=100,horizon=5180)
    assert_state_equal(before,train.capture_training_rng(torch.device("cpu")))
    assert_state_equal(state,m.state_dict())
    assert m.training and all(p.grad is None for p in m.parameters())
    assert result["norms"]["pressure_pair/raw"]["H"]["raw"] > 0
    assert result["cosines"]["supervised_motion/raw:pressure_pair/raw"]["H"] is None


def test_routed_trainer_resume_preserves_updates_exposure_rng_and_snapshots(tmp_path):
    groups = ToyGroups(cameras=(4,4,4,4))
    common = {**training_config(tmp_path), **cfg(), "effective_batch_rgb_windows":16,
              "microbatch_rgb_windows":8,"target_updates":4,"target_rgb_windows":64,
              "gradient_probe_updates":[2,4],"snapshot_updates":[2,4], "save_initial_checkpoint":True}
    def run(where,stop=None,resume=None):
        torch.manual_seed(15)
        return train.train_visual_run({**common,"control_dir":str(where/"control")},where/"run",resume=resume,
                                      stop_after_updates=stop,dataset=groups,validation_dataset=groups,
                                      model=TinyRoute(),validation_fn=validation)
    full=tmp_path/"full"
    split=tmp_path/"split"
    run(full)
    run(split,stop=2)
    run(split,resume=split/"run/last.pt")
    a,b=(torch.load(p/"run/last.pt",weights_only=False) for p in (full,split))
    for key in ("model","optimizer","scheduler","scaler","sampler","rng"):
        assert_state_equal(a[key],b[key])
    assert a["training"]["exposure_ledger"] == b["training"]["exposure_ledger"]
    assert sum(a["training"]["exposure_ledger"]["per_camera"].values()) == 64
    assert (full/"run/snapshots/step_000002.pt").is_file()
    assert (full/"run/gradient_probes/step_000004.json").is_file()


def test_unequal_valid_counts_accumulate_full_update_denominators_for_routed_losses():
    groups = ToyGroups(cameras=(4,2,3,1))
    items=[groups[i] for i in range(4)]
    items[0]["masks"]["pressure_valid"][:3,:48]=False
    items[2]["masks"]["motion_valid"][4:8]=False
    batches=[grouped_collate(items[:2]),grouped_collate(items[2:])]
    full=grouped_collate(items)
    config={**loss_config(),**cfg(),"tw":20,"grad_clip":1e6}
    torch.manual_seed(121)
    a=TinyRoute(dropout=0)
    b=copy.deepcopy(a)
    def step(model,parts):
        opt=torch.optim.SGD(model.parameters(),lr=1e-3)
        return train.visual_accumulation_step(model,parts,config,opt,torch.amp.GradScaler("cuda",enabled=False),
                                             torch.device("cpu"),520,5180)
    x=step(a,[full])
    y=step(b,batches)
    assert x["counts"] == y["counts"]
    for name in x["raw"]:
        assert x["raw"][name] == pytest.approx(y["raw"][name],rel=2e-6,abs=1e-7)
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-5,atol=1e-7)
        torch.testing.assert_close(pa,pb,rtol=2e-6,atol=1e-8)
    assert a.forward_calls == 1 and b.forward_calls == 2


def test_cpu_amp_scaler_backoff_replays_identical_batch_and_dropout_for_routing():
    # CPU GradScaler exercises real found_inf/unscale behavior without CUDA.
    # Inject one overflow in an allowed pressure parameter's scaled gradient.
    groups=ToyGroups(cameras=(4,4))
    batch=grouped_collate([groups[0],groups[1]])
    config={**loss_config(),**cfg(),"tw":20,"grad_clip":1e6}
    torch.manual_seed(991)
    m=TinyRoute(dropout=.2)
    reference=copy.deepcopy(m)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    refopt=torch.optim.Adam(reference.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler("cpu",enabled=True,init_scale=128.,growth_interval=2000)
    refscaler=torch.amp.GradScaler("cpu",enabled=True,init_scale=64.,growth_interval=2000)
    triggered=[]
    def overflow_once(grad):
        if not triggered:
            triggered.append(True)
            return torch.full_like(grad,float("inf"))
        return grad
    hook=m.aux_heads.t_rec.weight.register_hook(overflow_once)
    state={"attempt_steps":0,"successful_updates":0,"rgb_windows":0,"last_sample_keys":list(batch["meta"]["sample_key"])}
    torch.manual_seed(772)
    rng=train.capture_training_rng(torch.device("cpu"))
    result=train.visual_update_with_amp_retries(m,[batch],config,opt,scaler,torch.device("cpu"),520,5180,state)
    hook.remove()
    train.restore_training_rng(rng,torch.device("cpu"))
    expected=train.visual_accumulation_step(reference,[batch],config,refopt,refscaler,torch.device("cpu"),520,5180)
    assert result["stepped"] and expected["stepped"]
    assert state["attempt_steps"] == 2 and state["amp_overflow_attempts"] == 1
    assert state["rgb_windows"] == 0 and state["successful_updates"] == 0
    assert scaler.get_scale() == 64.
    for pa,pb in zip(m.parameters(),reference.parameters()):
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    assert_state_equal(opt.state_dict(),refopt.state_dict())
    assert m.forward_calls == 2 and reference.forward_calls == 1


def test_large_gradient_norm_and_cosine_use_accurate_reductions_with_zero_na():
    # Three million differently sized coordinates reproduce the long-vector
    # precision problem observed in full-model F derivatives, without CUDA.
    class LargeGroup(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion=nn.Module()
            self.fusion.weight=nn.Parameter(torch.zeros(3_159_040))
            self.aux_heads=nn.Module()
            self.aux_heads.t_rec=nn.Linear(1,1)
    model=LargeGroup()
    generator=torch.Generator().manual_seed(232)
    direction=torch.randn(model.fusion.weight.numel(),generator=generator)*torch.logspace(-8,0,model.fusion.weight.numel())
    loss=(model.fusion.weight*direction).sum()
    result=gradient_statistics({'raw':loss,'weighted':.1*loss,'zero':0.*loss},model)
    independent_norm=float(direction.double().numpy() @ direction.double().numpy())**.5
    assert result['norms']['raw']['F']['raw'] == pytest.approx(independent_norm,rel=1e-11)
    assert result['cosines']['raw:weighted']['F'] == pytest.approx(1.,rel=0,abs=1e-12)
    assert -1 <= result['cosines']['raw:weighted']['F'] <= 1
    assert result['cosines']['raw:zero']['F'] is None
    assert result['cosines']['raw:weighted']['H'] is None
    assert all(p.grad is None for p in model.parameters())


def test_measurement_replicas_get_distinct_identity_and_are_forbidden_formal():
    base=cfg(stage='pilot')
    first={**base,'measurement_replica_id':'cohortA/gpu0/slot0'}
    second={**base,'measurement_replica_id':'cohortA/gpu0/slot1'}
    validate_a2_config(first)
    validate_a2_config(second)
    assert train.visual_run_identity(first) != train.visual_run_identity(second)
    assert train.visual_run_identity(first) != train.visual_run_identity(base)
    unchanged=copy.deepcopy(base)
    unchanged.pop('stage')
    assert 'measurement_replica_id' not in unchanged
    for value in ('',None,12):
        with pytest.raises(ValueError,match='measurement_replica_id'):
            validate_a2_config({**base,'measurement_replica_id':value})
    with pytest.raises(ValueError,match='measurement_replica_id'):
        validate_a2_config({**first,'stage':'formal'},formal=True)


def a3_config(run="R-HF-VT40", seed=1, **extra):
    experiment, head, pose, pressure, pair, supervised = A3_RUNS[run]
    return {"a2_version": A3_VERSION, "a2_run_id": run, "seed": seed,
            "a2_head_seed": seed, "experiment": experiment, "pressure_head": head,
            "query_memory": "fusion_tactile_slice" if run.endswith("-T20") else "fusion_all",
            "pressure_pair_route": pair, "supervised_pressure_route": supervised,
            "lambda_mv": .2 if pose or pressure else 0., "mv_pose_weight": pose/.2, "mv_pressure_weight": pressure/.2,
            "effective_pose_pair_peak": pose, "effective_pressure_pair_peak": pressure,
            "budget_mode": "a3_extended", "stage": "formal", "target_updates": 10360,
            "effective_batch_rgb_windows": 252, "microbatch_rgb_windows": 6,
            "target_rgb_windows": 10360*252, "snapshot_updates": [5180, 10360],
            "train_cameras": [1,2,4], "validation_cameras": [1,2,4],
            "selection_motion_weight": .8, "selection_pressure_weight": .2,
            "selection_m_ref": .1, "selection_p_ref": .2, "save_initial_checkpoint": True,
            "pressure_cell_mean": [.1]*96, "pressure_contact_rate": [.3,.7],
            "pressure_contact_threshold": [.1,.1], "pressure_contact_pos_weight": [1.,1.],
            "lambda_contact": .01, "pressure_train_statistics_hash": "train-only-fixture", **extra}


@pytest.mark.parametrize("run", list(A3_RUNS))
@pytest.mark.parametrize("seed", [1,2,3])
def test_a3_complete_configuration_matrix(run, seed):
    config = a3_config(run, seed)
    validate_a2_config(config, formal=True)
    train._validate_visual_experiment(config)
    assert config["pressure_pair_route"] == "HF"
    assert not (config["effective_pose_pair_peak"] and config["effective_pressure_pair_peak"])


def test_a3_formal_config_supports_read_only_preflight(tmp_path):
    import yaml
    from anysole.types import validate_view_config
    template = yaml.safe_load((Path(__file__).resolve().parents[1]/"anysole/configs/view_consistency.yaml").read_text())
    config = {**template, **a3_config()}
    validate_view_config(config, formal=False)
    for key in ("manifest_path", "hrnet_checkpoint", "hrnet_config", "detector_checkpoint", "smpl_model_path"):
        config[key] = str(tmp_path/("missing-"+key))
    report = train.visual_preflight(config)
    assert report["ready"] is False and report["errors"]
    assert all("stage" not in error for error in report["errors"])
    with pytest.raises(ValueError, match="stage"):
        validate_a2_config({**a3_config(), "stage":"pilot"}, formal=True)


@pytest.mark.parametrize("override", [
    {"seed": 4}, {"seed": True}, {"a2_head_seed": 2}, {"a2_run_id": "Q-FACT"},
    {"pressure_head": "query_direct"}, {"query_memory": "fusion_visual_slice"},
    {"pressure_pair_route": "all"}, {"supervised_pressure_route": "all"},
    {"mv_pose_weight": 1.}, {"effective_pressure_pair_peak": .1},
    {"train_cameras": [1,2,3,4]}, {"validation_cameras": [3]},
    {"selection_motion_weight": .5}, {"selection_pressure_weight": .5},
    {"target_updates": 5180, "target_rgb_windows": 5180*252},
    {"target_rgb_windows": 1}, {"effective_batch_rgb_windows": 256},
    {"microbatch_rgb_windows": 3}, {"microbatch_rgb_windows": 8}, {"microbatch_rgb_windows": 255},
    {"snapshot_updates": [10360,5180]}, {"snapshot_updates": [5180]},
    {"selection_m_ref": float("nan")}, {"pressure_cell_mean": [0.]*96},
    {"measurement_replica_id": "formal-replica"}, {"pilot_schedule_update_offset": 100},
])
def test_a3_rejects_scientific_contract_drift(override):
    with pytest.raises(ValueError):
        validate_a2_config(a3_config(**override), formal=True)


@pytest.mark.parametrize("memory", ["fusion_tactile_slice", "fusion_all"])
def test_a3_query_memory_masks_gradients_and_no_extra_capacity(memory):
    torch.manual_seed(17)
    kwargs = dict(kind="query_factorized", dim=32, nhead=4, hidden=16,
                  dropout=0, cell_mean=[.1]*96, contact_rate=[.3,.7])
    head = QueryPressureHead(**kwargs, memory=memory).eval()
    reference = QueryPressureHead(**kwargs)
    assert sum(p.numel() for p in head.parameters()) == sum(p.numel() for p in reference.parameters())
    # Initial Q-FACT weights intentionally predict train means. Open this path
    # before testing memory dependence; otherwise a leak test is vacuous.
    with torch.no_grad():
        nn.init.normal_(head.output.weight, std=.1)
        nn.init.normal_(head.total.weight, std=.1)
    valid = torch.tensor([[True,False,True,True], [False]*4])
    time = torch.arange(4, dtype=torch.float64)[None].expand(2,-1)/40
    fused = torch.randn(2,8,32, requires_grad=True)
    seen = []
    hook = head.attention.register_forward_pre_hook(
        lambda module,args,kwargs: seen.append((args[1].shape[1],kwargs["key_padding_mask"].clone())),
        with_kwargs=True)
    original = head(fused, valid, time)["pressure_hat"]
    hook.remove()
    mask = torch.cat([valid[:1],valid[:1]],1) if memory == "fusion_all" else valid[:1]
    assert seen[0][0] == (8 if memory == "fusion_all" else 4)
    assert torch.equal(seen[0][1], ~mask)
    changed = fused.detach().clone()
    changed[:,[1,5]] = float("nan")
    changed[1] = float("nan")
    if memory == "fusion_tactile_slice":
        changed[:,:4] = float("nan")
    torch.testing.assert_close(head(changed, valid, time)["pressure_hat"], original, rtol=0, atol=0)
    original[valid].square().sum().backward()
    assert torch.isfinite(fused.grad).all()
    assert torch.count_nonzero(fused.grad[1]) == 0
    assert torch.count_nonzero(fused.grad[:,[1,5]]) == 0
    assert fused.grad[0,4:].abs().sum() > 0
    assert (fused.grad[0,:4].abs().sum() > 0) == (memory == "fusion_all")


def test_a3_memory_identity_and_full_model_reload(tmp_path):
    configs = [a3_config("R-HF-"+suffix, dropout=0) for suffix in ("T20", "VT40")]
    models = [visual.build(c).eval() for c in configs]
    assert models[0].structure_hash != models[1].structure_hash
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    for model in models:
        path = tmp_path / (model.pressure_readout.memory+".pt")
        torch.save({"model":model.state_dict(),"config":model.effective_config},path)
        saved = torch.load(path, weights_only=True)
        restored = visual.build(saved["config"]).eval()
        restored.load_state_dict(saved["model"],strict=True)
        v = torch.randn(1,20,2051)
        valid = torch.ones(1,20,dtype=torch.bool)
        valid[:,3] = False
        v[:,3] = float("nan")
        times = torch.arange(20,dtype=torch.float64)[None]/40
        with torch.no_grad():
            a,b = model.forward_visual(v,valid,times),restored.forward_visual(v,valid,times)
        for name in ("pressure_hat","pressure_total","pressure_contact_logits","x0_hat","v_hat"):
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
            assert torch.isfinite(a[name]).all()
        wrong = "fusion_all" if model.pressure_readout.memory == "fusion_tactile_slice" else "fusion_tactile_slice"
        with pytest.raises(ValueError,match="structure_hash"):
            visual.build({**saved["config"],"query_memory":wrong})


@pytest.mark.parametrize("name", ["pressure", "pressure_contact", "pressure_total", "pressure_diff"])
def test_a3_all_gt_pressure_terms_obey_the_supervised_route(name):
    for route in ("all","HF"):
        model = TinyRoute(dropout=0)
        out = model.forward_visual(torch.randn(2,5,2051),torch.ones(2,5,dtype=torch.bool),None)
        routed_backward({name:out["pressure_hat"].square().mean()},model,
                        {"supervised_pressure_route":route},torch.amp.GradScaler("cuda",enabled=False))
        norms = {k:sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
                 for k,ps in parameter_groups(model).items()}
        assert norms["H"] > 0 and norms["F"] > 0
        assert (norms["R"] > 0) == (route == "all")
        assert (norms["E_shared"] > 0) == (route == "all")
