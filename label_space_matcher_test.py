import os
from gaiavision.label_space.matchers import GivenLabelsMatcher
from gaiavision.label_space.matchers import Word2vecMatcher

DATA_ROOT = "unittest/examples/label_space"


def given_labels_matcher_test():
    matcher = GivenLabelsMatcher(
        source=os.path.join(DATA_ROOT, "given_labels_input.csv"),
        target=os.path.join(DATA_ROOT, "uni.0.0.3.csv"),
        dataset_name="kitti",
        )
    matcher.match()
    matcher.to_csv(os.path.join(DATA_ROOT, "given_labels_output.csv"))


def word2vec_matcher_test():
    matcher = Word2vecMatcher(
        source=os.path.join(DATA_ROOT, "given_labels_input.csv"),
        target=os.path.join(DATA_ROOT, "uni.0.0.3.csv"),
        dataset_name="kitti",
        sim_model="word2vec-google-news-300",
        sim_threshold=0.5,
        num_recommend=5,
        )
    matcher.match()
    matcher.to_csv(os.path.join(DATA_ROOT, "word2vec_output.csv"))


if __name__ == "__main__":
    given_labels_matcher_test()
    word2vec_matcher_test()
