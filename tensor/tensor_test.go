package tensor_test

import (
	"testing"

	"github.com/blast-go/blast/tensor"
)

func TestInvalidDimension(t *testing.T) {
	_, err := tensor.Zeros[int]([]uint{1, 0})
	if err == nil {
		t.Errorf("%s: should have failed due to invalid dimension", t.Name())
	}
}
func TestZerosTensor(t *testing.T) {
	t1, err := tensor.Zeros[int]([]uint{2, 2})
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}

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
	t1, err := tensor.Ones[float32]([]uint{2, 2})
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}

	for _, e := range t1.Elements() {
		if e != 1 {
			t.Errorf("%s: some element(s) are not 1", t.Name())
			break
		}
	}
}

func TestRandomTensor(t *testing.T) {
	t1, err := tensor.Rand[float32]([]uint{10})
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}
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
	t1, err := tensor.New([]uint{3, 2}, []uint8{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}
	expected := "[[1 2 3] [4 5 6]]"
	actual := t1.String()
	if actual != expected {
		t.Errorf("%s: expected=%s actual=%s", t.Name(), expected, actual)
	}
}

func TestTranspose(t *testing.T) {
	t1, err := tensor.Rand[float32]([]uint{3, 2})
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}

	t2, err := t1.Transpose()
	if err != nil {
		t.Errorf("%s: %v", t.Name(), err)
	}

	for i := uint(0); i < 3; i++ {
		for j := uint(0); j < 2; j++ {
			a, _ := t1.Get(i, j)
			b, _ := t2.Get(j, i)
			if a != b {
				t.Errorf("%s: elements not transposed correctly", t.Name())
			}
		}
	}
}
