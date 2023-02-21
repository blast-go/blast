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
// specified by the caller. Returns an error if the number of elements provided
// does not match the number of elements corresponding to its shape.
func New[T constraints.Number](shape []uint, elements []T) (*Tensor[T], error) {
	size := 1
	for i, d := range shape {
		if d == 0 {
			return nil, fmt.Errorf("dimension %d can't be zero", i)
		}
		size *= int(d)
	}

	if len(elements) != size {
		return nil, fmt.Errorf("invalid number of elements expected=%d got=%d", size, len(elements))
	}

	return &Tensor[T]{elements: elements, shape: shape}, nil
}

// Zeros returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are set to zero. If any dimension is set
// to zero the function will return an error.
func Zeros[T constraints.Number](shape []uint) (*Tensor[T], error) {
	size := 1
	for i, d := range shape {
		if d == 0 {
			return nil, fmt.Errorf("dimension %d can't be zero", i)
		}
		size *= int(d)
	}

	return &Tensor[T]{elements: make([]T, size), shape: shape}, nil
}

// Ones returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are set to one. If any dimension is set
// to zero the function will return an error.
func Ones[T constraints.Number](shape []uint) (*Tensor[T], error) {
	t, err := Zeros[T](shape)
	if err != nil {
		return nil, err
	}

	one := T(1)
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = one
	}

	return t, nil
}

// Rand returns a new Tensor of the shape and numeric type specified by the
// caller in which all its elements are random numbers. For integer types
// (uintX, intX) the full range is used, for floating point types (floatX)
// the generated numbers are in the range zero to one. If any dimension is set
// to zero the function will return an error.
func Rand[T constraints.Number](shape []uint) (*Tensor[T], error) {
	t, err := Zeros[T](shape)
	if err != nil {
		return nil, err
	}

	randFunc := randFuncFor(T(0))
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = randFunc()
	}

	return t, nil
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
// the tensor has more than two dimensions would return an error.
func (t *Tensor[T]) Transpose() (*Tensor[T], error) {
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("transpose only work for two dimensional tensors")
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

	return &Tensor[T]{elements: elements, shape: shape}, nil
}

// Get returns a single element located at the coordinates provided by the
// caller. Returns an error if the coordinates does not match the shape of
// the tensor.
func (t *Tensor[T]) Get(cords ...uint) (T, error) {
	if len(cords) != len(t.shape) {
		return T(0), fmt.Errorf("coordinates do not match the shape of the tensor")
	}

	offset := uint(0)
	prevSize := uint(1)
	for i, size := range t.shape {
		cord := cords[i]
		if cord > size {
			return T(0), fmt.Errorf("index at of bounds %d for size %d", cord, size)
		}
		offset += cord * prevSize
		prevSize = size
	}
	return t.elements[offset], nil
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
