import pytest
import pandas as pd
import numpy as np

from models.utils import create_sequences, accumulate_sequences, create_divisors

@pytest.fixture(scope="module")
def df():
    data = pd.DataFrame(
        dict(
            a = [1, 1, 1, 1, 1, 2,2,2,2,2], 
            b = [1,2,3,4,5,6,7,8,9,10],
            c = ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8","1-9","1-10"])
        )
    return data

@pytest.fixture(scope="module")
def result():
    sequences = [np.array([1,2,3]),
                np.array([2,3,4]),
                np.array([3,4,5]),
                np.array([6,7,8]),
                np.array([7,8,9]),
                np.array([8,9,10])]

    return sequences

@pytest.fixture(scope="module")
def acc_seq():
    datos = pd.DataFrame(dict(a = [1,2,3], b = [2,3,4], c = [3,4,5])).to_numpy()
    return datos


def test_create_sequences(df, result):
    sequences, idxs = create_sequences(df, "a", "b", "c", seq_len = 3)
    assert len(sequences) == 6
    assert len(idxs) == 6

    np.testing.assert_equal(np.array(sequences), np.array(result))

def test_accumulate_sequences(acc_seq, result):
    np.testing.assert_equal(accumulate_sequences(acc_seq, 5, 1, 3), np.array([1.0,4.0,9.0,8.0,5.0]))
    np.testing.assert_equal(accumulate_sequences(result, 5, 2, 3), np.array([1.0,4.0,9.0,8.0,5.0,6.0,14.0,24.0,18.0,10.0]))

def test_create_divisors():
    assert create_divisors(5, 1, 3) == [1,2,3,2,1]
    assert create_divisors(7, 1, 3) == [1, 2, 3, 3, 3, 2, 1]
    assert create_divisors(9, 1, 3) == [1, 2, 3, 3, 3, 3, 3, 2, 1]
