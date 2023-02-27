package device

import (
	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/tensor"
)

type Device[T constraints.Number] interface {
	Add(*tensor.Tensor[T], *tensor.Tensor[T]) *tensor.Tensor[T]
	Sub(*tensor.Tensor[T], *tensor.Tensor[T]) *tensor.Tensor[T]
	MatMul(*tensor.Tensor[T], *tensor.Tensor[T]) *tensor.Tensor[T]
}
