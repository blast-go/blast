package cpu

import (
	"math"

	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/tensor"
)

// CPU device performs computations using the processor's ALU instead of a
// dedicated device like a GPU.
type CPU[T constraints.Number] struct {
	grad bool
}

type option func(*options)

type options struct {
	grad bool
}

// This option enables the sum of gradients to be caluclated in a backward pass.
func WithGrad(v bool) option {
	return func(o *options) {
		o.grad = v
	}
}

var defaultOptions *options = &options{
	grad: false,
}

func New[T constraints.Number](opts ...option) CPU[T] {
	c := CPU[T]{}

	cfg := defaultOptions
	for _, o := range opts {
		o(cfg)
	}

	c.grad = cfg.grad

	return c
}

// Returns a new Tensor that is the result of adding the two tensors on element
// by element. Panics if the two tensors do not have the same shape.
func (c CPU[T]) Add(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	if !tensor.EqualShape(t1, t2) {
		panic("tensors must have the same shape")
	}

	shape := t1.Shape()

	forward := func() []T {
		t1Elements := t1.Elements()
		t2Elements := t2.Elements()
		size := len(t1Elements)
		elements := make([]T, size)
		for i := 0; i < size; i++ {
			elements[i] = t1Elements[i] + t2Elements[i]
		}
		return elements
	}

	parents := []*tensor.Tensor[T]{t1, t2}
	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(t *tensor.Tensor[T]) {
			t1Grad := t1.Grad()
			t2Grad := t2.Grad()
			grad := t.Grad()
			for i, g := range grad {
				t1Grad[i] += g
				t2Grad[i] += g
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Returns a new Tensor that is the result of subtracting the two tensors on
// element by element. Panics if the two tensors do not have the same shape.
func (c CPU[T]) Sub(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	if !tensor.EqualShape(t1, t2) {
		panic("tensors must have the same shape")
	}

	shape := t1.Shape()

	forward := func() []T {

		t1Elements := t1.Elements()
		t2Elements := t2.Elements()
		size := len(t1Elements)
		elements := make([]T, size)
		for i := 0; i < size; i++ {
			elements[i] = t1Elements[i] - t2Elements[i]
		}
		return elements
	}

	parents := []*tensor.Tensor[T]{t1, t2}
	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(t *tensor.Tensor[T]) {
			t1Grad := t1.Grad()
			t2Grad := t2.Grad()
			grad := t.Grad()
			for i, g := range grad {
				t1Grad[i] += g
				t2Grad[i] -= g
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Returns a new Tensor that is the result of matrix multiplication of the two
// input tensors. Panics if the shape of the two tensors is incompatible or if
// any of the input tensors are of order different than 2.
func (c CPU[T]) MatMul(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	if len(t1.Shape()) != 2 || len(t2.Shape()) != 2 {
		panic("cannot do matrix multiplication on tensor of higher order")
	}

	shape := tensor.Shape{t2.Shape()[0], t1.Shape()[1]}
	t1Shape := t1.Shape()
	t2Shape := t2.Shape()
	w1 := t1Shape[0]
	h1 := t1Shape[1]
	w2 := t2Shape[0]
	h2 := t2Shape[1]

	forward := func() []T {

		m1 := t1.Elements()
		m2 := transpose(t2.Elements(), w2, h2)
		m := make([]T, w2*h1)
		matmul(m, m1, m2, w2, h1, w1)
		return m
	}

	parents := []*tensor.Tensor[T]{t1, t2}
	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(t *tensor.Tensor[T]) {
			tGrad := t.Grad()

			//dL/dA = dL/dC @ B^T
			t1Grad := t1.Grad()
			t2Elements := t2.Elements()
			matmul(t1Grad, tGrad, t2Elements, h2, h1, w2)

			//dL/dB = A^T @ dL/dC
			tTGrad := transpose(tGrad, w2, h1)
			t1TElements := transpose(t1.Elements(), w1, h1)
			t2Grad := t2.Grad()
			matmul(t2Grad, t1TElements, tTGrad, h1, w1, h1)
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Transpose returns a new transposed Tensor over the first two dimensions. If
// the tensor has more than two dimensions would panic.
func (c CPU[T]) Transpose(t *tensor.Tensor[T]) *tensor.Tensor[T] {
	oldShape := t.Shape()

	if len(oldShape) != 2 {
		panic("transpose only work for two dimensional tensors")
	}

	w := oldShape[0]
	h := oldShape[1]
	shape := tensor.Shape{h, w}
	parents := []*tensor.Tensor[T]{t}

	forward := func() []T {
		tElements := t.Elements()
		tTElements := transpose(tElements, w, h)

		for i := uint(0); i < w; i++ {
			for j := uint(0); j < h; j++ {
				tTElements[j+i*h] = tElements[i+j*w]
			}
		}
		return tTElements
	}

	var backward func(*tensor.Tensor[T])
	if c.grad {
		backward = func(tT *tensor.Tensor[T]) {
			tTGrad := transpose(tT.Grad(), h, w)
			tGrad := t.Grad()
			for i, g := range tTGrad {
				tGrad[i] += g
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Tanh returns a new tensor with tanh activation function applied element-wise.
func (c CPU[T]) Tanh(t *tensor.Tensor[T]) *tensor.Tensor[T] {
	shape := t.Shape()
	parents := []*tensor.Tensor[T]{t}

	forward := func() []T {
		tElements := t.Elements()
		elements := make([]T, len(tElements))
		for i, e := range tElements {
			elements[i] = T(math.Tanh(float64(e)))
		}
		return elements
	}

	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(tout *tensor.Tensor[T]) {
			toutGrad := tout.Grad()
			tElements := t.Elements()
			tGrad := t.Grad()
			for i, g := range toutGrad {
				tanh := math.Tanh(float64(tElements[i]))
				tGrad[i] += T(1-tanh*tanh) * g
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Sigmoid returns a new tensor with sigmoid activation function applied
// element-wise.
func (c CPU[T]) Sigmoid(t *tensor.Tensor[T]) *tensor.Tensor[T] {
	shape := t.Shape()
	parents := []*tensor.Tensor[T]{t}

	forward := func() []T {
		tElements := t.Elements()
		elements := make([]T, len(tElements))
		for i, e := range tElements {
			elements[i] = T(1 / (1 + math.Exp(-float64(e))))
		}
		return elements
	}

	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(tOut *tensor.Tensor[T]) {
			tOutGrad := tOut.Grad()
			tOutElements := tOut.Elements()
			tGrad := t.Grad()
			for i, g := range tOutGrad {
				tGrad[i] += g * tOutElements[i] * (1 - tOutElements[i])
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// ReLU returns a new tensor with ReLU activation function applied element-wise.
func (c CPU[T]) ReLU(t *tensor.Tensor[T]) *tensor.Tensor[T] {
	shape := t.Shape()
	parents := []*tensor.Tensor[T]{t}

	forward := func() []T {
		tElements := t.Elements()
		elements := make([]T, len(tElements))
		for i, e := range tElements {
			if e > 0 {
				elements[i] = e
			}
		}
		return elements
	}

	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(tout *tensor.Tensor[T]) {
			toutElements := tout.Elements()
			toutGrad := tout.Grad()
			tGrad := t.Grad()
			for i, g := range toutGrad {
				if toutElements[i] > 0 {
					tGrad[i] += g
				}
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// PowInt returns a new tensor in which each element is raised to the power
// of exp which is an unsigned integer.
func (c CPU[T]) PowInt(t *tensor.Tensor[T], exp uint) *tensor.Tensor[T] {
	shape := t.Shape()
	parents := []*tensor.Tensor[T]{t}

	forward := func() []T {
		tElements := t.Elements()
		elements := make([]T, len(tElements))
		for i, e := range tElements {
			elements[i] = powInt(e, exp)
		}
		return elements
	}

	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(tOut *tensor.Tensor[T]) {
			tOutGrad := tOut.Grad()
			tGrad := t.Grad()
			tElements := t.Elements()
			for i, g := range tOutGrad {
				tGrad[i] += g * T(exp) * powInt(tElements[i], exp-1)
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

// Mul returns a new tensor multiplied by scale element-wise.
func (c CPU[T]) Mul(t *tensor.Tensor[T], scale T) *tensor.Tensor[T] {
	shape := t.Shape()
	parents := []*tensor.Tensor[T]{t}
	forward := func() []T {
		tElements := t.Elements()
		elements := make([]T, len(tElements))
		for i, e := range tElements {
			elements[i] = e * scale
		}
		return elements
	}
	var backward tensor.BackwardFunc[T]
	if c.grad {
		backward = func(tOut *tensor.Tensor[T]) {
			tOutGrad := tOut.Grad()
			tGrad := t.Grad()

			for i, g := range tOutGrad {
				tGrad[i] += g * scale
			}
		}
	}

	return tensor.Op(shape, parents, forward, backward)
}

func powInt[T constraints.Number](n T, exponent uint) T {
	switch exponent {
	case 0:
		return 1
	case 1:
		return n
	default:
		r := n
		for j := uint(1); j < exponent; j++ {
			r *= n
		}
		return r
	}
}

func transpose[T constraints.Number](elements []T, w, h uint) []T {
	tElements := make([]T, len(elements))

	for i := uint(0); i < w; i++ {
		for j := uint(0); j < h; j++ {
			tElements[j+i*h] = elements[i+j*w]
		}
	}
	return tElements
}

func matmul[T constraints.Number](m, m1, m2 []T, w, h, l uint) {
	// this function expect m2 to be transposed
	// w and h are width and height of the output matrix
	// l is the common dimension
	for i := uint(0); i < h; i++ {
		for j := uint(0); j < w; j++ {
			for k := uint(0); k < l; k++ {
				m[i*w+j] += m1[i*l+k] * m2[j*l+k]
			}
		}
	}
}
