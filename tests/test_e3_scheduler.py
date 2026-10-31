"""The real shell queue claims each task once and preserves failed exit codes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class E3SchedulerTests(unittest.TestCase):
    def test_parallel_claim_failure_and_isolated_state(self):
        with tempfile.TemporaryDirectory(dir='/private/Tactile/.ai/tmp') as directory:
            root=Path(directory)
            shutil.copy(Path(__file__).resolve().parents[1]/'configs/scheduler.sh',root/'scheduler.sh')
            (root/'run.sh').write_text('#!/usr/bin/env bash\nname=$(basename "$1")\nprintf "%s\\n" "$name" >> "$ANYSOLE_QUEUE_STATE_ROOT/executed"\n[[ "$name" != *bad* ]] || exit 7\nsleep 0.1\n')
            state=root/'isolated';queue=state/'queue';queue.mkdir(parents=True)
            for name in ('01good.conf','02bad.conf','03good.conf','04good.conf'):
                (queue/name).write_text('fixture\n')
            env=dict(os.environ,ANYSOLE_QUEUE_STATE_ROOT=str(state))
            workers=[subprocess.Popen(['bash',str(root/'scheduler.sh'),str(queue)],env=env,
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(3)]
            for worker in workers:
                out,err=worker.communicate(timeout=20)
                self.assertEqual(worker.returncode,1,(out,err))
            executed=(state/'executed').read_text().splitlines()
            self.assertEqual(len(executed),4);self.assertEqual(len(set(executed)),4)
            self.assertEqual(len(list((state/'done').glob('*.conf'))),3)
            self.assertEqual((state/'failed/02bad.conf.exit_code').read_text().strip(),'7')
            self.assertFalse(list((state/'running').glob('*.conf')))
            self.assertFalse((root/'running').exists())


if __name__=='__main__': unittest.main()
