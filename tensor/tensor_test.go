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

func TestBackward(t *testing.T) {
	t1 := tensor.Rand[float32](tensor.Shape{2, 2})
	t2 := tensor.Rand[float32](tensor.Shape{2, 2})
	t3 := tensor.Op(
		tensor.Shape{2, 2},
		[]*tensor.Tensor[float32]{t1, t2},
		nil, func(tout *tensor.Tensor[float32]) {
			grad := tout.Grad()
			t1Grad := t1.Grad()
			t2Grad := t2.Grad()

			for i, g := range grad {
				t1Grad[i] += g * 0.5
				t2Grad[i] -= g * 1.5
			}
		})
	t3.Backward()
	all := true
	for _, g := range t1.Grad() {
		if g != 0.5 {
			all = false
		}
	}
	if !all {
		t.Errorf("%s: error applying gradients", t.Name())
	}

	all = true
	for _, g := range t2.Grad() {
		if g != -1.5 {
			all = false
		}
	}
	if !all {
		t.Errorf("%s: error applying gradients", t.Name())
	}

}

func TestEqualShape(t *testing.T) {
	t1 := tensor.Empty[uint16](tensor.Shape{2, 20})
	t2 := tensor.Empty[uint16](tensor.Shape{2, 20})
	t3 := tensor.Empty[uint16](tensor.Shape{20, 2})

	if !tensor.EqualShape(t1, t2) {
		t.Errorf("%s: tensor do have equal shape", t.Name())
	}

	if tensor.EqualShape(t1, t3) {
		t.Errorf("%s: tensor do not have equal shape", t.Name())
	}

}

func TestEqual(t *testing.T) {
	t1 := tensor.New(tensor.Shape{2, 3}, []uint16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New(tensor.Shape{2, 3}, []uint16{1, 2, 3, 4, 5, 6})
	t3 := tensor.New(tensor.Shape{3, 2}, []uint16{1, 2, 3, 4, 5, 6})
	t4 := tensor.New(tensor.Shape{2, 3}, []uint16{0, 2, 3, 4, 5, 6})

	if !tensor.Equal(t1, t2) {
		t.Errorf("%s: tensor do have equal shape", t.Name())
	}

	if tensor.Equal(t1, t3) {
		t.Errorf("%s: tensor do not have equal shape", t.Name())
	}

	if tensor.Equal(t1, t4) {
		t.Errorf("%s: tensor do not have equal shape", t.Name())
	}

}
