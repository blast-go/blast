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

func NewTensor[T constraints.Number](dimensions ...uint) *Tensor[T] {
	size := uint(1)

	if len(dimensions) != 0 {
		for _, d := range dimensions {
			size *= d
		}
	} else {
		dimensions = []uint{1}
	}

	return &Tensor[T]{
		elements: make([]T, size),
		shape:    dimensions,
	}
}

func (t *Tensor[T]) Rand() {
	randFunc := randFuncFor(t)
	for i := 0; i < len(t.elements); i++ {
		t.elements[i] = randFunc()
	}
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

func randFuncFor[T constraints.Number](t *Tensor[T]) func() T {
	switch reflect.ValueOf(T(0)).Kind() {
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
