package tests

import (
	"reflect"
	"testing"

	"github.com/retzl4ff/cletron/pkg/matrix"
)

func TestDotProductResult(t *testing.T) {
	mt1 := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	mt2 := [][]float64{
		{1},
		{2},
		{3},
		{4},
	}	

	newMatrix, _ := matrix.DotProduct(mt1, mt2)
	 
	want := [][]float64{
		{30},
		{70},
		{110},
	}

	if !reflect.DeepEqual(newMatrix, want) {
		t.Errorf("expected %v; got %v", want, newMatrix)
	}
}

func TestWrongDimensions(t *testing.T) {
	mt1 := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	mt2 := [][]float64{
		{1},
		{2},
		{3},
	}	

	_, err := matrix.DotProduct(mt1, mt2)
	
	want := err != nil

	if !want {
		t.Errorf("Expected err to be not nil")
	}
}

func TestMatrixSum(t *testing.T) {
	m1 := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	m2 := [][]float64{
		{-1, -2, -3, -4},
		{-5, -6, -7, -8},
		{-9, -10, -11, -12},
	}	

	newMatrix, _ := matrix.MatrixSum(m1, m2)
	
	want := [][]float64{
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	}	

	if !reflect.DeepEqual(newMatrix, want) {
		t.Errorf("expected %v; got %v", want, newMatrix)
	}
}

func TestMultiplyByScalar(t *testing.T) {
	m := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	scalar := 0.0
	
	newMatrix, _ := matrix.MultiplyByScalar(m, scalar)
	
	want := [][]float64{
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	}


	if !reflect.DeepEqual(newMatrix, want) {
		t.Errorf("expected %v; got %v", want, newMatrix)
	}
}