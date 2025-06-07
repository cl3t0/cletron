package matrix

import (
	"errors"
)

func DotProduct(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	result := make([][]float64, len(m1))

	m1_row_quantity := len(m1)
	m1_column_quantity := len(m1[0])
	m2_row_quantity := len(m2)
	m2_column_quantity := len(m2[0])

	if m1_column_quantity != m2_row_quantity {
		return nil, errors.New("the number of columns in the first matrix must be equal to the number of rows in the second matrix")
	}

	for i := 0; i < m1_row_quantity; i++ {
		row := make([]float64, m2_column_quantity)

		for j := 0; j < m2_column_quantity; j++ {
			sum := 0.0

			for k := 0; k < len(m2); k++ {
				m1_element := m1[i][k]
				m2_element := m2[k][j]

				sum += m1_element * m2_element
			}

			row[j] = sum
		}

		result[i] = row
	}

	return result, nil
}

func MatrixSum(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	rowQuantity := len(m1)
	columnQuantity := len(m1[0])

	if rowQuantity != len(m2) {
		return nil, errors.New("row quantity does not match")
	}

	if columnQuantity != len(m2[0]) {
		return nil, errors.New("column quantity does not match")
	}

	result := make([][]float64, rowQuantity)

	for i := 0; i < rowQuantity; i++ {
		row := make([]float64, columnQuantity)

		for j := 0; j < columnQuantity; j++ {
			row[j] = m1[i][j] + m2[i][j]
		}

		result[i] = row
	}

	return result, nil
}

func MultiplyByScalar(m [][]float64, scalar float64) ([][]float64, error) {
	rowQuantity := len(m)
	columnQuantity := len(m[0])

	result := make([][]float64, rowQuantity)

	for i := 0; i < rowQuantity; i++ {
		row := make([]float64, columnQuantity)

		for j := 0; j < columnQuantity; j++ {
			row[j] = m[i][j] * scalar
		}

		result[i] = row
	}

	return result, nil
}

func Transpose(m [][]float64) [][]float64 {
	rowQuantity := len(m)
	columnQuantity := len(m[0])
	result := make([][]float64, columnQuantity)

	for i := 0; i < len(result); i++ {
		result[i] = make([]float64, rowQuantity)
	}

	for i := 0; i < rowQuantity; i++ {
		for j := 0; j < columnQuantity; j++ {
			result[j][i] = m[i][j]
		}
	}

	return result
}
