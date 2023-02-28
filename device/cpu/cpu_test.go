package cpu_test

import (
	"testing"

	"github.com/blast-go/blast/device/cpu"
	"github.com/blast-go/blast/tensor"
)

// FIXME: test for backward pass
func TestAdd(t *testing.T) {
	d := cpu.New[uint16]()
	t1 := tensor.New([]uint{2, 2, 2}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{2, 2, 2}, []uint16{8, 7, 6, 5, 4, 3, 2, 1})

	if !tensor.Equal(d.Add(t1, t2), tensor.New(tensor.Shape{2, 2, 2}, []uint16{9, 9, 9, 9, 9, 9, 9, 9})) {
		t.Errorf("%s: Add failed", t.Name())
	}
}

// FIXME: test for backward pass
func TestSub(t *testing.T) {
	d := cpu.New[int64]()
	t1 := tensor.New([]uint{1, 8}, []int64{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{1, 8}, []int64{8, 7, 6, 5, 4, 3, 2, 1})

	if !tensor.Equal(d.Sub(t1, t2), tensor.New(tensor.Shape{1, 8}, []int64{-7, -5, -3, -1, 1, 3, 5, 7})) {
		t.Errorf("%s: Sub failed", t.Name())
	}
}

func TestMatMul2D(t *testing.T) {
	d := cpu.New[int16]()
	t1 := tensor.New([]uint{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New([]uint{2, 3}, []int16{7, 8, 9, 10, 11, 12})

	if !tensor.Equal(d.MatMul(t1, t2), tensor.New(tensor.Shape{2, 2}, []int16{58, 64, 139, 154})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}

	if !tensor.Equal(d.MatMul(t2, t1), tensor.New(tensor.Shape{3, 3}, []int16{39, 54, 69, 49, 68, 87, 59, 82, 105})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}

	t3 := tensor.New([]uint{1, 2}, []int16{1, 4})
	if !tensor.Equal(d.MatMul(t2, t3), tensor.New(tensor.Shape{1, 3}, []int16{39, 49, 59})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}
}

func TestMatMulGrad2D(t *testing.T) {
	d := cpu.New[int16](cpu.WithGrad(true))
	t1 := tensor.New([]uint{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New([]uint{2, 3}, []int16{7, 8, 9, 10, 11, 12})
	t3 := d.MatMul(t1, t2)
	t3.Backward()

	expected := []int16{15, 19, 23, 15, 19, 23}
	for i, g := range t1.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}

	expected = []int16{5, 5, 7, 7, 9, 9}
	for i, g := range t2.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}
}

func TestTranspose(t *testing.T) {
	d := cpu.New[float32]()

	t1 := tensor.Rand[float32]([]uint{3, 2})
	t2 := d.Transpose(t1)

	for i := uint(0); i < 3; i++ {
		for j := uint(0); j < 2; j++ {
			if t1.Get(i, j) != t2.Get(j, i) {
				t.Errorf("%s: transpose failed", t.Name())
			}
		}
	}
}
