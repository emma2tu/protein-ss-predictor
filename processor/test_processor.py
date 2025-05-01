import tempfile
import pytest
from processor.processor import ProteinDataProcessor
import config

config.residue_13_dict = {aa: aa for aa in "ACDEFGHIKLMNPQRSTVWY"}  # mock 1-letter to 3-letter map

@pytest.fixture
def sample_data():
    train = "id\tsecondary_structure\nP1_ALA_1\tH\nP1_GLY_2\tE\n"
    test = "id\nP1_ALA_1\nP1_GLY_2\n"
    fasta = ">P1\nAG\n"
    return train, test, fasta

def test_pipeline(sample_data):
    train, test, fasta = sample_data
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_train, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_test, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_fasta:
        
        f_train.write(train); f_train.flush()
        f_test.write(test); f_test.flush()
        f_fasta.write(fasta); f_fasta.flush()

        processor = ProteinDataProcessor()
        processor.process_all(f_train.name, f_test.name, f_fasta.name)

        assert 'P1' in processor.training_set or 'P1' in processor.validation_set
        assert 'P1' in processor.testing_set
        assert isinstance(processor.training_set.get('P1') or processor.validation_set.get('P1'), tuple)
        assert len(processor.raw_sequences_dict['P1']) == 2

def test_invalid_residues_are_filtered():
    train = "id\tsecondary_structure\nP1_ALA_1\tH\n"
    test = "id\nP1_ALA_1\n"
    fasta = ">P1\nAXZ\n"  # 'X' and 'Z' are not valid amino acids

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_train, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_test, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_fasta:
        
        f_train.write(train); f_train.flush()
        f_test.write(test); f_test.flush()
        f_fasta.write(fasta); f_fasta.flush()

        processor = ProteinDataProcessor()
        processor.process_all(f_train.name, f_test.name, f_fasta.name)

        assert 'P1' not in processor.training_set
        assert 'P1' not in processor.validation_set
        assert 'P1' not in processor.testing_set

def test_missing_labels_are_skipped():
    train = "id\tsecondary_structure\nP1_ALA_1\tH\nP1_GLY_2\t\n"  # Missing label for second residue
    test = "id\nP1_ALA_1\n"
    fasta = ">P1\nAG\n"

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_train, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_test, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_fasta:
        
        f_train.write(train); f_train.flush()
        f_test.write(test); f_test.flush()
        f_fasta.write(fasta); f_fasta.flush()

        processor = ProteinDataProcessor()
        processor.process_all(f_train.name, f_test.name, f_fasta.name)

        # Should not crash and still process first residue
        assert 'P1' in processor.training_set or 'P1' in processor.validation_set

def test_sequence_shorter_than_labels_is_filtered():
    train = "id\tsecondary_structure\nP1_ALA_1\tH\nP1_GLY_2\tE\nP1_SER_3\tT\n"
    test = "id\nP1_ALA_1\n"
    fasta = ">P1\nAG\n"  # Only 2 residues, not enough for 3 labels

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_train, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_test, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_fasta:
        
        f_train.write(train); f_train.flush()
        f_test.write(test); f_test.flush()
        f_fasta.write(fasta); f_fasta.flush()

        processor = ProteinDataProcessor()
        processor.process_all(f_train.name, f_test.name, f_fasta.name)

        assert 'P1' not in processor.training_sequences_dict  # Skipped for being too short


