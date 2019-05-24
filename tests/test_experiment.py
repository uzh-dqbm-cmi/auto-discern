import unittest
import autodiscern.experiment as ade


class TestExperiment(unittest.TestCase):

    def test_partition_document_ids_all_elements_preserved(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)

        input_set = set(doc_ids)
        output_set = set([item for sublist in output for item in sublist])
        self.assertEqual(input_set, output_set)

    def test_partition_document_ids_elements_do_not_overlap(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)

        # assert that the elements in each sublist are not duplicated in other sublists
        for i in range(len(output)):
            for j in range(len(output)):
                if i != j:
                    setA = set(output[i])
                    setB = set(output[j])
                    intersection = setA.intersection(setB)
                    self.assertEqual(len(intersection), 0)

    def test_partition_document_ids_total_number_of_elements_preserved(self):
        doc_ids = list(range(20))
        output = ade.PartitionedExperiment.partition_document_ids(doc_ids, 4)
        self.assertEqual(sum([len(sublist) for sublist in output]), len(doc_ids))

    def test_partition_document_ids_stratified_all_elements_preserved(self):
        doc_ids = list(range(20))
        labels = [1]*10 + [2]*10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)

        input_set = set(doc_ids)
        output_set = set([item for sublist in output for item in sublist])
        self.assertEqual(input_set, output_set)

    def test_partition_document_ids_stratified_elements_do_not_overlap(self):
        doc_ids = list(range(20))
        labels = [1] * 10 + [2] * 10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)

        # assert that the elements in each sublist are not duplicated in other sublists
        for i in range(len(output)):
            for j in range(len(output)):
                if i != j:
                    setA = set(output[i])
                    setB = set(output[j])
                    intersection = setA.intersection(setB)
                    self.assertEqual(len(intersection), 0)

    def test_partition_document_ids_stratified_total_number_of_elements_preserved(self):
        doc_ids = list(range(20))
        labels = [1] * 10 + [2] * 10
        output = ade.PartitionedExperiment.partition_document_ids_stratified(doc_ids, labels, 4)
        self.assertEqual(sum([len(sublist) for sublist in output]), len(doc_ids))

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
