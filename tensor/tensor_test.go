package tensor_test

import (
	"testing"

	"github.com/blast-go/blast/tensor"
)

func TestCreateTensor(t *testing.T) {
	t1 := tensor.NewTensor[int](2, 2)
	actual := len(t1.Elements())
	if actual != 4 {
		t.Errorf("%s: incorrect number of elements actual=%d", t.Name(), actual)
	}
}

func TestRandomTensor(t *testing.T) {
	t1 := tensor.NewTensor[float32](10)
	t1.Rand()
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
	t1 := tensor.NewTensor[uint8](3, 2)
	t1.SetElements([]uint8{1, 2, 3, 4, 5, 6})
	expected := "[[1 2 3] [4 5 6]]"
	actual := t1.String()
	if actual != expected {
		t.Errorf("%s: expected=%s actual=%s", t.Name(), expected, actual)
	}
}
