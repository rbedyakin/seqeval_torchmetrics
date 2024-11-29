import unittest
from metric.seqeval_metric import Seqeval
import torch
from torch import tensor


class TestMetric(unittest.TestCase):

    def AssertResultClose(self, actual, expected, rtol=1e-4, atol=1e-4):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for k, v in expected.items():
            torch.testing.assert_close(actual=actual[k],
                                       expected=v,
                                       rtol=rtol,
                                       atol=atol)

    def test_1(self):
        metric = Seqeval(labels=["MISC", "PER"])

        predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'],
                       ['B-PER', 'I-PER', 'O']]
        references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'],
                      ['B-PER', 'I-PER', 'O']]

        actual = metric(predictions, references)
        expected = {
            'MISC_precision': tensor(0.),
            'PER_precision': tensor(1.),
            'MISC_recall': tensor(0.),
            'PER_recall': tensor(1.),
            'MISC_f1': tensor(0.),
            'PER_f1': tensor(1.),
            'MISC_number': tensor(1),
            'PER_number': tensor(1),
            'overall_precision': tensor(0.5000),
            'overall_recall': tensor(0.5000),
            'overall_f1': tensor(0.5000)
        }
        self.AssertResultClose(actual, expected)

    def test_2(self):
        """
        Evaluation test is performed for the following dataset.
        https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        
        To reproduce:
            1) Download https://www.clips.uantwerpen.be/conll2000/chunking/output.txt.gz
            2) Exctact files from archive
            3) Move `output.txt` to `./data/`

        """

        def load_test_output(path: str):
            with open(path) as inp:
                references = []
                predictions = []
                reference = []
                prediction = []
                for line in inp:
                    row = line.strip()
                    if row:
                        token, pos, y_true, y_pred = row.split()
                        reference.append(y_true)
                        prediction.append(y_pred)
                    else:
                        references.append(reference)
                        predictions.append(prediction)
                        reference = []
                        prediction = []
            if reference:
                references.append(reference)
                predictions.append(prediction)

            return references, predictions

        metric = Seqeval(labels=["ADJP", "ADVP", "NP", "PP", "SBAR", "VP"])

        references, predictions = load_test_output(path='./data/output.txt')

        actual = metric(predictions, references)
        expected = {
            'ADJP_precision': tensor(0.),
            'ADVP_precision': tensor(0.4545),
            'NP_precision': tensor(0.6498),
            'PP_precision': tensor(0.8318),
            'SBAR_precision': tensor(0.6667),
            'VP_precision': tensor(0.6900),
            'ADJP_recall': tensor(0.),
            'ADVP_recall': tensor(0.6250),
            'NP_recall': tensor(0.7863),
            'PP_recall': tensor(0.9889),
            'SBAR_recall': tensor(0.3333),
            'VP_recall': tensor(0.7931),
            'ADJP_f1': tensor(0.),
            'ADVP_f1': tensor(0.5263),
            'NP_f1': tensor(0.7116),
            'PP_f1': tensor(0.9036),
            'SBAR_f1': tensor(0.4444),
            'VP_f1': tensor(0.7380),
            'ADJP_number': tensor(6),
            'ADVP_number': tensor(8),
            'NP_number': tensor(262),
            'PP_number': tensor(90),
            'SBAR_number': tensor(6),
            'VP_number': tensor(87),
            'overall_precision': tensor(0.6883),
            'overall_recall': tensor(0.8083),
            'overall_f1': tensor(0.7435)
        }
        self.AssertResultClose(actual, expected)

    def test_3(self):
        metric = Seqeval(labels=["NP"])

        references = [['B-NP', 'I-NP', 'O']]
        predictions = [['I-NP', 'I-NP', 'O']]

        actual = metric(predictions, references)
        expected = {
            'NP_precision': tensor(1.),
            'NP_recall': tensor(1.),
            'NP_f1': tensor(1.),
            'NP_number': tensor(1),
            'overall_precision': tensor(1.),
            'overall_recall': tensor(1.),
            'overall_f1': tensor(1.)
        }
        self.AssertResultClose(actual, expected)

    def test_4(self):
        metric = Seqeval(labels=["NP"], mode='strict', scheme='IOB2')

        references = [['B-NP', 'I-NP', 'O']]
        predictions = [['I-NP', 'I-NP', 'O']]

        actual = metric(predictions, references)

        expected = {
            'NP_precision': tensor(0.),
            'NP_recall': tensor(0.),
            'NP_f1': tensor(0.),
            'NP_number': tensor(1),
            'overall_precision': tensor(0.),
            'overall_recall': tensor(0.),
            'overall_f1': tensor(0.)
        }
        self.AssertResultClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
