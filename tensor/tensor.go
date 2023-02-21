package tensor

import (
	"fmt"
	"math/rand"
	"reflect"

	"github.com/blast-go/blast/constraints"
)

type Tensor[T constraints.Number] struct {
	elements []T
	shape    []uint
}

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

func (t *Tensor[T]) Elements() []T {
	return t.elements
}

func (t *Tensor[T]) SetElements(elements []T) error {
	if len(elements) != len(t.elements) {
		return fmt.Errorf("size mismatch expected=%d got=%d", len(t.elements), len(elements))
	}
	t.elements = elements
	return nil
}

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
