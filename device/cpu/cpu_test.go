package cpu_test

import (
	"math"
	"testing"

	"github.com/blast-go/blast/device/cpu"
	"github.com/blast-go/blast/tensor"
)

// FIXME: test for backward pass
func TestAdd(t *testing.T) {
	d := cpu.New[uint16]()
	t1 := tensor.New(tensor.Shape{2, 2, 2}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New(tensor.Shape{2, 2, 2}, []uint16{8, 7, 6, 5, 4, 3, 2, 1})

	if !tensor.Equal(d.Add(t1, t2), tensor.New(tensor.Shape{2, 2, 2}, []uint16{9, 9, 9, 9, 9, 9, 9, 9})) {
		t.Errorf("%s: Add failed", t.Name())
	}
}

func TestAddGrad(t *testing.T) {
	d := cpu.New[uint16](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{1, 8}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New(tensor.Shape{1, 8}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t3 := d.Add(t1, t2)
	t3.Backward()

	expected := []uint16{1, 1, 1, 1, 1, 1, 1, 1}
	for i, g := range t1.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}

	for i, g := range t2.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}
}

func TestSub(t *testing.T) {
	d := cpu.New[int64]()
	t1 := tensor.New(tensor.Shape{1, 8}, []int64{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New(tensor.Shape{1, 8}, []int64{8, 7, 6, 5, 4, 3, 2, 1})

	if !tensor.Equal(d.Sub(t1, t2), tensor.New(tensor.Shape{1, 8}, []int64{-7, -5, -3, -1, 1, 3, 5, 7})) {
		t.Errorf("%s: Sub failed", t.Name())
	}
}

func TestSubGrad(t *testing.T) {
	d := cpu.New[int16](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{1, 8}, []int16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New(tensor.Shape{1, 8}, []int16{1, 2, 3, 4, 5, 6, 7, 8})
	t3 := d.Sub(t1, t2)
	t3.Backward()

	expected := []int16{1, 1, 1, 1, 1, 1, 1, 1}
	for i, g := range t1.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}

	expected = []int16{-1, -1, -1, -1, -1, -1, -1, -1}
	for i, g := range t2.Grad() {
		if g != expected[i] {
			t.Errorf("%s: gradient failed. expected=%d got=%d", t.Name(), expected[i], g)
		}
	}
}

func TestMatMul2D(t *testing.T) {
	d := cpu.New[int16]()
	t1 := tensor.New(tensor.Shape{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New(tensor.Shape{2, 3}, []int16{7, 8, 9, 10, 11, 12})

	if !tensor.Equal(d.MatMul(t1, t2), tensor.New(tensor.Shape{2, 2}, []int16{58, 64, 139, 154})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}

	if !tensor.Equal(d.MatMul(t2, t1), tensor.New(tensor.Shape{3, 3}, []int16{39, 54, 69, 49, 68, 87, 59, 82, 105})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}

	t3 := tensor.New(tensor.Shape{1, 2}, []int16{1, 4})
	if !tensor.Equal(d.MatMul(t2, t3), tensor.New(tensor.Shape{1, 3}, []int16{39, 49, 59})) {
		t.Errorf("%s: MatMul failed", t.Name())
	}
}

func TestMatMulGrad2D(t *testing.T) {
	d := cpu.New[int16](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New(tensor.Shape{2, 3}, []int16{7, 8, 9, 10, 11, 12})
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

	t1 := tensor.Rand[float32](tensor.Shape{3, 2})
	t2 := d.Transpose(t1)

	for i := uint(0); i < 3; i++ {
		for j := uint(0); j < 2; j++ {
			if t1.Get(i, j) != t2.Get(j, i) {
				t.Errorf("%s: transpose failed", t.Name())
			}
		}
	}
}

func TestTransposeGrad(t *testing.T) {
	//FIXME: implement
}

func TestSigmoid(t *testing.T) {
	d := cpu.New[float32]()
	t1 := tensor.New(tensor.Shape{3, 3}, []float32{-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100})
	t2 := d.Sigmoid(t1)
	expected := tensor.New(
		tensor.Shape{3, 3},
		[]float32{3.8e-44, 4.539787e-05, 0.26894143, 0.47502083, 0.5, 0.5249792, 0.7310586, 0.9999546, 1},
	)

	if !tensor.Equal(t2, expected) {
		t.Errorf("%s: Sigmoid failed expected=%v got=%v", t.Name(), expected, t2)
	}
}

func TestSigmoidGrad(t *testing.T) {
	d := cpu.New[float32](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 3}, []float32{-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100})
	t2 := d.Sigmoid(t1)
	t2.Backward()

	expected := []float32{3.8e-44, 4.539581e-05, 0.19661194, 0.24937604, 0.25, 0.24937604, 0.19661193, 4.5397937e-05, 0}
	maxDelta := 0.001
	for i, g := range t1.Grad() {
		if math.Abs(float64(g)-float64(expected[i])) > maxDelta {
			t.Errorf("%s: Sigmoid grad failed expected=%f got=%f", t.Name(), expected[i], g)
		}
	}
}

func TestPowInt(t *testing.T) {
	d := cpu.New[int](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 3}, []int{-100, -10, -2, -1, 0, 1, 2, 10, 100})
	t2 := d.PowInt(t1, 2)

	expected := tensor.New(tensor.Shape{3, 3}, []int{10000, 100, 4, 1, 0, 1, 4, 100, 10000})
	if !tensor.Equal(t2, expected) {
		t.Errorf("%s: PowInt failed expected=%v got=%v", t.Name(), expected, t2)
	}
}

func TestPowIntGrad(t *testing.T) {
	d := cpu.New[int](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 3}, []int{-100, -10, -2, -1, 0, 1, 2, 10, 100})
	t2 := d.PowInt(t1, 2)
	t2.Backward()

	expected := []int{-200, -20, -4, -2, 0, 2, 4, 20, 200}

	for i, g := range t1.Grad() {
		if g != expected[i] {
			t.Errorf("%s: PowInt grad failed expected=%d got=%d", t.Name(), expected[i], g)
		}
	}
}

func TestMul(t *testing.T) {
	d := cpu.New[float32](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 3}, []float32{-100, -10, -2, -1, 0, 1, 2, 10, 100})
	t2 := d.Mul(t1, 0.5)

	expected := tensor.New(tensor.Shape{3, 3}, []float32{-50, -5, -1, -0.5, 0, 0.5, 1, 5, 50})
	if !tensor.Equal(t2, expected) {
		t.Errorf("%s: Mul failed expected=%v got=%v", t.Name(), expected, t2)
	}
}

func TestMulGrad(t *testing.T) {
	d := cpu.New[float64](cpu.WithGrad(true))
	t1 := tensor.New(tensor.Shape{3, 3}, []float64{-100, -10, -2, -1, 0, 1, 2, 10, 100})
	t2 := d.Mul(t1, 0.5)
	t2.Backward()

	expected := []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	for i, g := range t1.Grad() {
		if g != expected[i] {
			t.Errorf("%s: Mul grad failed expected=%f got=%f", t.Name(), expected[i], g)
		}
	}
}
