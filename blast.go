package main

import (
	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/device"
	"github.com/blast-go/blast/device/cpu"
	"github.com/blast-go/blast/tensor"
)

type Engine[T constraints.Number] struct {
	device device.Device[T]
}

func New[T constraints.Number]() *Engine[T] {
	device := cpu.New[T]()
	return &Engine[T]{device: device}
}

func (e *Engine[T]) Add(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	return e.device.Add(t1, t2)
}

func (e *Engine[T]) Sub(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	return e.device.Sub(t1, t2)
}

func (e *Engine[T]) MatMul(t1, t2 *tensor.Tensor[T]) *tensor.Tensor[T] {
	return e.device.MatMul(t1, t2)
}
