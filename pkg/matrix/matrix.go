package matrix

import (
	"errors"
)

func DotProduct(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	result := make([][]float64, len(m1))

	m1RowQuantity := len(m1)
	m1ColumnQuantity := len(m1[0])
	m2RowQuantity := len(m2)
	m2ColumnQuantity := len(m2[0])

	if m1ColumnQuantity != m2RowQuantity {
		return nil, errors.New("the number of columns in the first matrix must be equal to the number of rows in the second matrix")
	}

	for i := range m1RowQuantity {
		row := make([]float64, m2ColumnQuantity)

		for j := range m2ColumnQuantity {
			sum := 0.0

			for k := range len(m2) {
				m1Element := m1[i][k]
				m2Element := m2[k][j]

				sum += m1Element * m2Element
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

	for i := range rowQuantity {
		row := make([]float64, columnQuantity)

		for j := range columnQuantity {
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

	for i := range rowQuantity {
		row := make([]float64, columnQuantity)

		for j := range columnQuantity {
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

	for i := range result {
		result[i] = make([]float64, rowQuantity)
	}

	for i := range rowQuantity {
		for j := range columnQuantity {
			result[j][i] = m[i][j]
		}
	}

	return result
}
