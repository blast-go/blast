package tensor

import (
	"fmt"
	"math/rand"
	"reflect"

	"github.com/blast-go/blast/constraints"
)

// Tensor is the basic type that stores values over N dimensions.
type Tensor[T constraints.Number] struct {
	elements []T
	shape    []uint
}

// New returns a new Tensor of the shape, numeric type and the elements are
// specified by the caller. Panics if the number of elements provided
// does not match the number of elements corresponding to its shape.
func New[T constraints.Number](shape []uint, elements []T) *Tensor[T] {
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

// Zeros returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are set to zero. If any dimension is set
// to zero the function will panic.
func Zeros[T constraints.Number](shape []uint) *Tensor[T] {
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
func Ones[T constraints.Number](shape []uint) *Tensor[T] {
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
func Rand[T constraints.Number](shape []uint) *Tensor[T] {
	t := Zeros[T](shape)

	randFunc := randFuncFor(T(0))
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = randFunc()
	}

	return t
}

// Elements returns all elements of the tensor as a slice of the tensor type.
func (t *Tensor[T]) Elements() []T {
	return t.elements
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

// Transpose returns a new transposed Tensor over the first two dimensions. If
// the tensor has more than two dimensions would panic.
func (t *Tensor[T]) Transpose() *Tensor[T] {
	if len(t.shape) != 2 {
		panic("transpose only work for two dimensional tensors")
	}

	w := t.shape[0]
	h := t.shape[1]

	shape := []uint{h, w}
	elements := make([]T, len(t.elements))

	for i := uint(0); i < w; i++ {
		for j := uint(0); j < h; j++ {
			elements[j+i*h] = t.elements[i+j*w]
		}
	}

	return &Tensor[T]{elements: elements, shape: shape}
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
	return t.elements[offset]
}

// Returns a new Tensor that is the result of adding the two tensors on element
// by element. Panics if the two tensors do not have the same shape.
func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	if len(t.shape) != len(other.shape) {
		panic("tensors must have the same shape")
	}

	for i := 0; i < len(t.shape); i++ {
		if t.shape[i] != other.shape[i] {
			panic("tensors must have the same shape")
		}
	}

	shape := make([]uint, len(t.shape))
	copy(shape, t.shape)

	size := len(t.elements)
	elements := make([]T, size)
	for i := 0; i < size; i++ {
		elements[i] = t.elements[i] + other.elements[i]
	}
	return &Tensor[T]{elements: elements, shape: shape}
}

// Returns a new Tensor that is the result of subtracting the two tensors on
// element by element. Panics if the two tensors do not have the same shape.
func (t *Tensor[T]) Sub(other *Tensor[T]) *Tensor[T] {
	if len(t.shape) != len(other.shape) {
		panic("tensors must have the same shape")
	}

	for i := 0; i < len(t.shape); i++ {
		if t.shape[i] != other.shape[i] {
			panic("tensors must have the same shape")
		}
	}

	shape := make([]uint, len(t.shape))
	copy(shape, t.shape)

	size := len(t.elements)
	elements := make([]T, size)
	for i := 0; i < size; i++ {
		elements[i] = t.elements[i] - other.elements[i]
	}
	return &Tensor[T]{elements: elements, shape: shape}
}

// Returns a new Tensor that is the result of matrix multiplication of the two
// input tensors. Panics if the shape of the two tensors is incompatible or if
// any of the input tensors are of order different than 2.
func (t *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	if len(t.shape) != 2 || len(other.shape) != 2 {
		panic("cannot do matrix multiplication on tensor of higher order")
	}

	tT := other.Transpose()

	w := t.shape[0]
	h := t.shape[1]

	shape := []uint{h, h}
	elements := make([]T, h*h)

	for i := uint(0); i < h; i++ {
		for j := uint(0); j < h; j++ {
			for k := uint(0); k < w; k++ {
				elements[i*h+j] += t.elements[i*w+k] * tT.elements[j*w+k]
			}
		}
	}

	return &Tensor[T]{elements: elements, shape: shape}
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
