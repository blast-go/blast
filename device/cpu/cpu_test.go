package cpu_test

import (
	"fmt"
	"testing"

	"github.com/blast-go/blast/constraints"
	"github.com/blast-go/blast/device/cpu"
	"github.com/blast-go/blast/tensor"
)

func TestAdd(t *testing.T) {
	d := cpu.New[uint16]()
	t1 := tensor.New([]uint{2, 2, 2}, []uint16{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{2, 2, 2}, []uint16{8, 7, 6, 5, 4, 3, 2, 1})

	if ok, err := equal(d.Add(t1, t2), []uint16{9, 9, 9, 9, 9, 9, 9, 9}); !ok {
		t.Errorf("%s: Add failed: %v", t.Name(), err)
	}
}

func TestSub(t *testing.T) {
	d := cpu.New[int64]()
	t1 := tensor.New([]uint{1, 8}, []int64{1, 2, 3, 4, 5, 6, 7, 8})
	t2 := tensor.New([]uint{1, 8}, []int64{8, 7, 6, 5, 4, 3, 2, 1})

	if ok, err := equal(d.Sub(t1, t2), []int64{-7, -5, -3, -1, 1, 3, 5, 7}); !ok {
		t.Errorf("%s: Sub failed: %v", t.Name(), err)
	}
}

func TestMatMul2D(t *testing.T) {
	d := cpu.New[int16]()
	t1 := tensor.New([]uint{3, 2}, []int16{1, 2, 3, 4, 5, 6})
	t2 := tensor.New([]uint{2, 3}, []int16{7, 8, 9, 10, 11, 12})

	if ok, err := equal(d.MatMul(t1, t2), []int16{58, 64, 139, 154}); !ok {
		t.Errorf("%s: MatMul failed: %v", t.Name(), err)
	}

	if ok, err := equal(d.MatMul(t2, t1), []int16{39, 54, 69, 49, 68, 87, 59, 82, 105}); !ok {
		t.Errorf("%s: MatMul failed: %v", t.Name(), err)
	}

	t3 := tensor.New([]uint{1, 2}, []int16{1, 4})
	if ok, err := equal(d.MatMul(t2, t3), []int16{39, 49, 59}); !ok {
		t.Errorf("%s: MatMul failed: %v", t.Name(), err)
	}

}

func equal[T constraints.Number](t *tensor.Tensor[T], expected []T) (bool, error) {
	if len(t.Elements()) != len(expected) {
		return false, fmt.Errorf("length mismatch expected=%d got=%d", len(expected), len(t.Elements()))
	}

	for i := 0; i < len(expected); i++ {
		if t.Elements()[i] != expected[i] {
			return false, fmt.Errorf("element does not match expected=%v got=%v", expected[i], t.Elements()[i])
		}
	}
	return true, nil
}
