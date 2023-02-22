package tensor_test

import (
	"fmt"
	"testing"

	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/tensor"
)

func TestInvalidDimension(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: should have failed due to invalid dimension", t.Name())
		}
	}()
	tensor.Zeros[int]([]uint{1, 0})
}
func TestZerosTensor(t *testing.T) {
	t1 := tensor.Zeros[int]([]uint{2, 2})

	actual := len(t1.Elements())
	if actual != 4 {
		t.Errorf("%s: incorrect number of elements actual=%d", t.Name(), actual)
	}

	for _, e := range t1.Elements() {
		if e != 0 {
			t.Errorf("%s: some element(s) are not 1", t.Name())
			break
		}
	}
}

func TestOnesTensor(t *testing.T) {
	t1 := tensor.Ones[float32]([]uint{2, 2})

	for _, e := range t1.Elements() {
		if e != 1 {
			t.Errorf("%s: some element(s) are not 1", t.Name())
			break
		}
	}
}

func TestRandomTensor(t *testing.T) {
	t1 := tensor.Rand[float32]([]uint{10})

	all := t1.Elements()
	first := all[0]
	equal := true
	for _, e := range all[1:] {
		if first != e {
			equal = false
		}
	}
	if equal {
		t.Errorf("%s: numbers are not random %s", t.Name(), t1)
	}
}

func TestTensorToString(t *testing.T) {
	t1 := tensor.New([]uint{3, 2}, []uint8{1, 2, 3, 4, 5, 6})

	expected := "[[1 2 3] [4 5 6]]"
	actual := t1.String()
	if actual != expected {
		t.Errorf("%s: expected=%s actual=%s", t.Name(), expected, actual)
	}
}

func TestTranspose(t *testing.T) {
	t1 := tensor.Rand[float32]([]uint{3, 2})
	t2 := t1.Transpose()

	for i := uint(0); i < 3; i++ {
		for j := uint(0); j < 2; j++ {
			if t1.Get(i, j) != t2.Get(j, i) {
				t.Errorf("%s: transpose failed", t.Name())
			}
		}
	}
}

func TestAdd(t *testing.T) {
	t1 := tensor.New([]uint{2, 2, 2}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{2, 2, 2}, []uint16{8, 7, 6, 5, 4, 3, 2, 1})

	if ok, err := equal(t1.Add(t2), []uint16{9, 9, 9, 9, 9, 9, 9, 9}); !ok {
		t.Errorf("%s: Add failed: %v", t.Name(), err)
	}
}

func TestSub(t *testing.T) {
	t1 := tensor.New([]uint{1, 8}, []int64{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{1, 8}, []int64{8, 7, 6, 5, 4, 3, 2, 1})

	if ok, err := equal(t1.Sub(t2), []int64{-7, -5, -3, -1, 1, 3, 5, 7}); !ok {
		t.Errorf("%s: Sub failed: %v", t.Name(), err)
	}
}

func TestMatMul(t *testing.T) {
	t1 := tensor.New([]uint{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New([]uint{2, 3}, []int16{7, 8, 9, 10, 11, 12})

	if ok, err := equal(t1.MatMul(t2), []int16{58, 64, 139, 154}); !ok {
		t.Errorf("%s: MatMul failed: %v", t.Name(), err)
	}

	if ok, err := equal(t2.MatMul(t1), []int16{39, 54, 69, 49, 68, 87, 59, 82, 105}); !ok {
		t.Errorf("%s: MatMul failed: %v", t.Name(), err)
	}
}

func equal[T constraints.Number](t *tensor.Tensor[T], expected []T) (bool, error) {
	if len(t.Elements()) != len(expected) {
		return false, fmt.Errorf("length mismatch expected=%d got=%d", len(expected), len(t.Elements()))
	}

	for i := 0; i < len(expected); i++ {
		if t.Elements()[i] != expected[i] {
			return false, fmt.Errorf("element does not match expected=%v got=%v", expected[i], t.Elements()[i])
		}
	}
	return true, nil
}
