from hw3 import INPUT_FILES_DIR, DATASET_FILE_SOYBEAN
from parser.arffparser import parse_file_to_lines, determine_number_of_classes, extract_examples

expected_first_feature_line_soybean = "0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,1\
,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0"
expected_first_label_soybean = "diaporthe-stem-canker"

expected_last_feature_line_soybean = "0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,0,0,1,0,1,0,\
0,1,0,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0"

expected_last_label_soybean = "herbicide-injury"

expected_soybean_k = 19

expected_example_count_soybean = 683


def test_hw3_parser():

    file_lines = parse_file_to_lines(INPUT_FILES_DIR, DATASET_FILE_SOYBEAN)
    examples = extract_examples(file_lines)
    actual_k = determine_number_of_classes(file_lines)
    actual_example_count = len(examples)



    assert actual_example_count == expected_example_count_soybean, "wrong number of total examples {}" \
        .format(actual_example_count)
    assert actual_k == expected_soybean_k, "wrong number of classes"

    actual_first_features = examples[0].features
    actual_first_line_label = examples[0].label
    actual_last_line_features = examples[-1].features
    actual_last_line_label = examples[-1].label

    assert actual_first_features == expected_first_feature_line_soybean.split(","), "first line's feature not correct"
    assert actual_first_line_label == expected_first_label_soybean, "first line's label not correct"
    assert actual_last_line_features == expected_last_feature_line_soybean.split(","), "last line's feature not correct"
    assert actual_last_line_label == expected_last_label_soybean, "last line's label not correct"

