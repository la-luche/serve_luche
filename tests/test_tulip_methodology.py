"""Guard the clinical task/video join independently of heuristic predictions."""
import ast
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import evaluate_tulip_methodology as evaluation


class StagedLabels:
    def json(self, key):
        # Existing hand staging supplies participant membership only. Finger
        # tapping must join the separate raw index-tapping recording.
        if key == evaluation.HAND_LABELS:
            return {'splits': {
                'train': ['subject2_task11_camera3.mp4', 'subject2_task12_camera4.mp4'],
                'val': ['subject9_task11_camera3.mp4', 'subject9_task12_camera4.mp4'],
                'test': [],
            }}
        return {'splits': {'train': [], 'val': [], 'test': []}}


class TulipTaskMappingTests(unittest.TestCase):
    def check_partition(self, partition, subject):
        rows = [row for row in evaluation.build_examples(StagedLabels(), partition)
                if row.test_type == 'fingerTapping']
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.subject for row in rows}, {subject})
        expected = {
            'Finger tapping - Left hand': ('left', '9. Index_Finger_tapping_left/Camera3.json'),
            'Finger tapping - Right hand': ('right', '10. Index_Finger_tapping_right/Camera4.json'),
        }
        for row in rows:
            side, recording = expected[row.target_name]
            self.assertEqual(row.side, side)
            self.assertEqual(row.keypoints_key,
                             f'{evaluation.GENERATED_PREFIX}/tulip/Subject_{subject}/Subject_{subject}/{recording}')

    def test_train_membership_and_correct_exercise(self):
        self.check_partition('train', 2)

    def test_held_out_membership_and_correct_exercise(self):
        self.check_partition('val', 9)

    def test_generic_evaluator_uses_the_same_clinical_task(self):
        # Read the declarative task map without importing optional reporting
        # dependencies (pandas/boto3) or connecting to R2.
        tree = ast.parse((TOOLS / 'evaluate_r2_heuristics.py').read_text())
        mapping = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'TULIP_TASKS'
                               for target in node.targets))
        tasks = {task: specs for task, specs in mapping.items()
                 if any(test_type == 'fingerTapping' for _, test_type in specs)}
        self.assertEqual(tasks, {
            '9. Index_Finger_tapping_left': [('Finger tapping - Left hand', 'fingerTapping')],
            '10. Index_Finger_tapping_right': [('Finger tapping - Right hand', 'fingerTapping')],
        })


if __name__ == '__main__':
    unittest.main()
