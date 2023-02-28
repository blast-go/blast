package tensor

import (
	"fmt"
	"math/rand"
	"reflect"

	"github.com/blast-go/blast/constraints"
)

type BackwardFunc[T constraints.Number] func(*Tensor[T])
type ForwardFunc[T constraints.Number] func() []T

// Tensor is the basic type that stores values over N dimensions.
type Tensor[T constraints.Number] struct {
	shape    Shape
	elements []T
	grad     []T
	parents  []*Tensor[T]
	forward  ForwardFunc[T]
	backward BackwardFunc[T]
}

// Type to describe the shape of a tensor.
type Shape = []uint

// New returns a new Tensor of the shape, numeric type and the elements are
// specified by the caller. Panics if the number of elements provided
// does not match the number of elements corresponding to its shape.
func New[T constraints.Number](shape Shape, elements []T) *Tensor[T] {
	size := 1
	for i, d := range shape {
		if d == 0 {
			panic(fmt.Sprintf("dimension %d can't be zero", i))
		}
		size *= int(d)
	}

	if len(elements) != size {
		panic(fmt.Sprintf("invalid number of elements expected=%d got=%d", size, len(elements)))
	}

	return &Tensor[T]{elements: elements, shape: shape}
}

// Empty returns a new Tensor of the given shape specified by the caller but
// doesn't allocate memory for the elements. If any dimension is set to zero
// the function will panic.
func Empty[T constraints.Number](shape Shape) *Tensor[T] {
	for i, d := range shape {
		if d == 0 {
			panic(fmt.Sprintf("dimension %d can't be zero", i))
		}
	}
	return &Tensor[T]{shape: shape}
}

// Zeros returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are set to zero. If any dimension is set
// to zero the function will panic.
func Zeros[T constraints.Number](shape Shape) *Tensor[T] {
	size := 1
	for i, d := range shape {
		if d == 0 {
			panic(fmt.Sprintf("dimension %d can't be zero", i))
		}
		size *= int(d)
	}

	return &Tensor[T]{elements: make([]T, size), shape: shape}
}

// Ones returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are set to one. If any dimension is set
// to zero the function will panic.
func Ones[T constraints.Number](shape Shape) *Tensor[T] {
	t := Zeros[T](shape)

	one := T(1)
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = one
	}

	return t
}

// Rand returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are random numbers. For integer types
// (uintX, intX) the full range is used, for floating point types (floatX)
// the generated numbers are in the range zero to one. If any dimension is set
// to zero the function will panic.
func Rand[T constraints.Number](shape Shape) *Tensor[T] {
	t := Zeros[T](shape)

	randFunc := randFuncFor(T(0))
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = randFunc()
	}

	return t
}

func Op[T constraints.Number](shape Shape, parents []*Tensor[T], forward ForwardFunc[T], backward BackwardFunc[T]) *Tensor[T] {
	return &Tensor[T]{shape: shape, parents: parents, forward: forward, backward: backward}
}

// Elements returns all elements of the tensor as a slice of the tensor type.
func (t *Tensor[T]) Elements() []T {
	if t.elements == nil && t.forward != nil {
		t.elements = t.forward()
	}
	return t.elements
}

// Grad returns all gradients of the tensor as a slice of float64.
func (t *Tensor[T]) Grad() []T {
	if t.grad == nil {
		size := 1
		for _, s := range t.shape {
			size *= int(s)
		}
		t.grad = make([]T, size)
	}
	return t.grad
}

// Shape returns the shape of the tensor.
func (t *Tensor[T]) Shape() Shape {
	return t.shape
}

// Returns an string representing the current tensor.
func (t *Tensor[T]) String() string {
	var arr []any = make([]any, len(t.elements))

	for i := 0; i < len(t.elements); i++ {
		arr[i] = t.elements[i]
	}

	for d := 0; d < len(t.shape)-1; d++ {
		bucketSize := int(t.shape[d])
		buckets := make([]any, len(arr)/int(bucketSize))
		for i := 0; i < len(buckets); i++ {
			s := i * bucketSize
			bucket := arr[s : s+bucketSize]
			buckets[i] = bucket
		}
		arr = buckets
	}
	return fmt.Sprint(arr)
}

// Get returns a single element located at the coordinates provided by the
// caller. Panics if the coordinates are out of bounds.
func (t *Tensor[T]) Get(cords ...uint) T {
	if len(cords) != len(t.shape) {
		panic("coordinates out of not match the shape of the tensor")
	}

	offset := uint(0)
	prevSize := uint(1)
	for i, size := range t.shape {
		cord := cords[i]
		if cord > size {
			panic(fmt.Sprintf("index out of bounds %d for size %d", cord, size))
		}
		offset += cord * prevSize
		prevSize = size
	}
	return t.Elements()[offset]
}

// Returns true if the two tensors have the same shape and elements, returns
// false otherwise.
func Equal[T constraints.Number](t1, t2 *Tensor[T]) bool {
	if !EqualShape(t1, t2) {
		return false
	}

	t1Elements := t1.Elements()
	t2Elements := t2.Elements()
	for i := 0; i < len(t1Elements); i++ {
		if t1Elements[i] != t2Elements[i] {
			return false
		}
	}

	return true
}

// Backward performs a back propagation pass for the entire computation graph.
// This should be called on the output of node of a graph.
func (t *Tensor[T]) Backward() {
	visited := make(map[*Tensor[T]]struct{})
	grad := t.Grad()
	for i := 0; i < len(grad); i++ {
		grad[i] = 1
	}
	applyBackward(t, visited)
}

func applyBackward[T constraints.Number](t *Tensor[T], visited map[*Tensor[T]]struct{}) {
	if _, ok := visited[t]; !ok {
		visited[t] = struct{}{}
		if t.backward != nil {
			t.backward(t)
		}
		for _, p := range t.parents {
			applyBackward(p, visited)
		}
	}
}

// Returns true if the two tensors have the same shape, returns false otherwise.
func EqualShape[T constraints.Number](t1, t2 *Tensor[T]) bool {
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

func randFuncFor[T constraints.Number](zero T) func() T {
	switch reflect.ValueOf(zero).Kind() {
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint:
		return func() T { return T(rand.Uint32()) }
	case reflect.Uint64:
		return func() T { return T(rand.Uint64()) }
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int:
		return func() T { return T(rand.Int31()) }
	case reflect.Int64:
		return func() T { return T(rand.Int63()) }
	case reflect.Float32:
		return func() T { return T(rand.Float32()) }
	case reflect.Float64:
		return func() T { return T(rand.Float64()) }
	default:
		panic("Unsupported data type")
	}
}
