package tensor_test

import (
	"testing"

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
				t.Errorf("%s: elements not transposed correctly", t.Name())
			}
		}
	}
}
