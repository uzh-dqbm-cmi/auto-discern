import unittest
import autodiscern.experiment as ade


class TestExperiment(unittest.TestCase):

    def test_partition_document_ids_all_elements_preserved(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)

        input_set = set(doc_ids)
        output_set = set([item for partition_name in output for item in output[partition_name]])
        self.assertEqual(input_set, output_set)

    def test_partition_document_ids_elements_do_not_overlap(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)
        output_list_of_lists = [output[partition_name] for partition_name in output]

        # assert that the elements in each sublist are not duplicated in other sublists
        for i in range(len(output_list_of_lists)):
            for j in range(len(output_list_of_lists)):
                if i != j:
                    setA = set(output_list_of_lists[i])
                    setB = set(output_list_of_lists[j])
                    intersection = setA.intersection(setB)
                    self.assertEqual(len(intersection), 0)

    def test_partition_document_ids_total_number_of_elements_preserved(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)
        self.assertEqual(sum([len(output[partition_name]) for partition_name in output]), len(doc_ids))

    def test_partition_document_ids_stratified_all_elements_preserved(self):
        doc_ids = list(range(20))
        labels = [1]*10 + [2]*10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)

        input_set = set(doc_ids)
        output_set = set([item for partition_name in output for item in output[partition_name]])
        self.assertEqual(input_set, output_set)

    def test_partition_document_ids_stratified_elements_do_not_overlap(self):
        doc_ids = list(range(20))
        labels = [1] * 10 + [2] * 10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)
        output_list_of_lists = [output[partition_name] for partition_name in output]

        # assert that the elements in each sublist are not duplicated in other sublists
        for i in range(len(output_list_of_lists)):
            for j in range(len(output_list_of_lists)):
                if i != j:
                    setA = set(output_list_of_lists[i])
                    setB = set(output_list_of_lists[j])
                    intersection = setA.intersection(setB)
                    self.assertEqual(len(intersection), 0)

    def test_partition_document_ids_stratified_total_number_of_elements_preserved(self):
        doc_ids = list(range(20))
        labels = [1] * 10 + [2] * 10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)
        self.assertEqual(sum([len(output[partition_name]) for partition_name in output]), len(doc_ids))

    def test_partition_document_ids_by_category(self):
        doc_ids = list(range(20))
        doc_categories = [1] * 5 + [2] * 5 + [3] * 10
        category_key = {
            1: 'one',
            2: 'two',
            3: 'three',
        }
        expected_output = {
            'one': list(range(0, 5)),
            'two': list(range(5, 10)),
            'three': list(range(10, 20)),
        }
        output = ade.PartitionedExperiment.partition_document_ids_by_category(doc_ids, doc_categories, category_key)
        self.assertEqual(output, expected_output)

    def test_materialize_partition(self):
        partition_ids = [3, 4]
        data_dict = {
            0: {'entity_id': 0, 'content': 'words'},
            1: {'entity_id': 1, 'content': 'words'},
            2: {'entity_id': 2, 'content': 'words'},
            3: {'entity_id': 3, 'content': 'words'},
            4: {'entity_id': 4, 'content': 'words'},
        }
        expected_output = (
            [
                {'entity_id': 0, 'content': 'words'},
                {'entity_id': 1, 'content': 'words'},
                {'entity_id': 2, 'content': 'words'},

            ],
            [
                {'entity_id': 3, 'content': 'words'},
                {'entity_id': 4, 'content': 'words'},

            ],
        )
        self.assertEqual(ade.PartitionedExperiment.materialize_partition(partition_ids, data_dict), expected_output)
