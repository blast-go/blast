package cpu

import (
	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/tensor"
)

type CPU[T constraints.Number] struct{}

func New[T constraints.Number]() CPU[T] {
	return CPU[T]{}
}

// Returns a new Tensor that is the result of adding the two tensors on element
// by element. Panics if the two tensors do not have the same shape.
func (c CPU[T]) Add(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	if !equalShape(t1, t2) {
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
	return tensor.Op(shape, forward)
}

// Returns a new Tensor that is the result of subtracting the two tensors on
// element by element. Panics if the two tensors do not have the same shape.
func (c CPU[T]) Sub(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	if !equalShape(t1, t2) {
		panic("tensors must have the same shape")
	}

	shape := t1.Shape()

	t1Elements := t1.Elements()
	t2Elements := t2.Elements()
	size := len(t1Elements)
	elements := make([]T, size)
	for i := 0; i < size; i++ {
		elements[i] = t1Elements[i] - t2Elements[i]
	}
	return tensor.New(shape, elements)
}

// Returns a new Tensor that is the result of matrix multiplication of the two
// input tensors. Panics if the shape of the two tensors is incompatible or if
// any of the input tensors are of order different than 2.
func (c CPU[T]) MatMul(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	t1Shape := t1.Shape()
	t2Shape := t2.Shape()

	if len(t1Shape) != 2 || len(t2Shape) != 2 {
		panic("cannot do matrix multiplication on tensor of higher order")
	}

	tT := t2.Transpose()

	w1 := t1Shape[0]
	h1 := t1Shape[1]
	w2 := t2Shape[0]

	t1Elements := t1.Elements()
	tTElements := tT.Elements()

	shape := tensor.Shape{w2, h1}
	elements := make([]T, w2*h1)

	for i := uint(0); i < h1; i++ {
		for j := uint(0); j < w2; j++ {
			for k := uint(0); k < w1; k++ {
				elements[i*w2+j] += t1Elements[i*w1+k] * tTElements[j*w1+k]
			}
		}
	}

	return tensor.New(shape, elements)
}

func equalShape[T constraints.Number](t1, t2 *tensor.Tensor[T]) bool {
	t1Shape := t1.Shape()
	t2Shape := t2.Shape()
	if len(t1Shape) != len(t2Shape) {
		return false
	}

	for i := 0; i < len(t1Shape); i++ {
		if t1Shape[i] != t2Shape[i] {
			return false
		}
	}
	return true
}
