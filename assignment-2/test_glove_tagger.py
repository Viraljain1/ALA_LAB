"""
Unit tests (unittest).  Run:  python -m unittest -v test_glove_tagger.py
Set GLOVE_PATH if the model is not at assets/glove50/glove_50_fast.wordvectors.
"""

import unittest

import numpy as np

import glove_tagger as gt
from vec import Vec

TEXT = ("Our MAHE students presented their research on campus! "
        "Faculty from the medicine department joined. #Manipal zzqxwvk")


class SemanticTaggingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = gt.load_glove()
        cls.tokens = gt.preprocess_text(TEXT)
        cls.tag_names, cls.T = gt.build_tag_matrix(cls.model, gt.TAG_VOCAB)
        cls.in_vocab, cls.oov, cls.W = gt.build_text_matrix(cls.model, cls.tokens)
        cls.S_vector = np.array(gt.similarity_matrix_vector_class(cls.W, cls.T))
        cls.S_numpy = gt.similarity_matrix_numpy(cls.W, cls.T)

    def test_1_tag_matrix_is_20_by_50(self):
        self.assertEqual(self.T.shape, (20, 50))

    def test_2_text_matrix_is_n_by_50(self):
        n = len(self.in_vocab)
        self.assertGreater(n, 0)
        self.assertEqual(self.W.shape, (n, 50))

    def test_3_similarity_matrix_is_n_by_20(self):
        n = len(self.in_vocab)
        self.assertEqual(self.S_numpy.shape, (n, 20))
        self.assertEqual(self.S_vector.shape, (n, 20))

    def test_4_one_pair_vec_vs_numpy(self):
        i = self.in_vocab.index("students")
        j = self.tag_names.index("university")
        by_hand = Vec(map(float, self.model["students"])).cosine_similarity(
            Vec(map(float, self.model["university"])))
        self.assertAlmostEqual(self.S_numpy[i, j], by_hand, places=6)

    def test_4b_multiword_pair_vec_vs_numpy(self):
        i = self.in_vocab.index("medicine")
        j = self.tag_names.index("public health")
        tag_vec = Vec(map(float, self.T[j]))
        by_hand = Vec(map(float, self.model["medicine"])).cosine_similarity(tag_vec)
        self.assertAlmostEqual(self.S_numpy[i, j], by_hand, places=9)

    def test_5_allclose(self):
        self.assertTrue(np.allclose(self.S_vector, self.S_numpy, atol=gt.TOLERANCE))

    def test_6_oov_token_excluded_and_reported(self):
        self.assertIn("zzqxwvk", self.tokens)
        self.assertNotIn("zzqxwvk", self.in_vocab)
        self.assertIn("zzqxwvk", self.oov)
        self.assertEqual(self.W.shape[0], len(self.tokens) - len(self.oov))

    def test_7_tag_with_oov_component_raises(self):
        broken = list(gt.TAG_VOCAB)
        broken[-1] = "careers zzqxwvk"
        with self.assertRaises(ValueError):
            gt.build_tag_matrix(self.model, broken)

    def test_repetitions_are_kept(self):
        in_vocab, _, W = gt.build_text_matrix(self.model, ["campus", "alumni", "campus"])
        self.assertEqual(in_vocab, ["campus", "alumni", "campus"])
        self.assertTrue(np.array_equal(W[0], W[2]))

    def test_preprocessing_order(self):
        self.assertEqual(gt.preprocess_text("The #AI lab, at MIT-Manipal!"),
                         ["lab", "mit", "manipal"])

    def test_zero_row_rejected(self):
        with self.assertRaises(ValueError):
            gt.row_normalize(np.zeros((1, 50)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
