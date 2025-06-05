package matrix

import (
	"errors"
)

func DotProduct(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	if len(m1[0]) != len(m2) {
		return nil, errors.New("the number of columns in the first matrix must be equal to the number of rows in the second matrix")
	}

	result := make([][]float64, len(m1))

	println(result)

	for i := 0; i < len(m1); i++ {
		row := make([]float64, len(m2[0]))

		for j := 0; j < len(m2[i]); j++ {
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
	row_quantity := len(m1)
	column_quantity := len(m1[0])

	if row_quantity != len(m2) {
		return nil, errors.New("row quantity does not match")
	}
	if column_quantity != len(m2[0]) {
		return nil, errors.New("column quantity does not match")
	}

	result := make([][]float64, row_quantity)

	for i := 0; i < row_quantity; i++ {
		row := make([]float64, column_quantity)

		for j := 0; j < column_quantity; j++ {
			row[j] = m1[i][j] + m2[i][j]
		}

		result[i] = row
	}

	return result, nil
}

func MultiplyByScalar(m [][]float64, scalar float64) ([][]float64, error) {
	row_quantity := len(m)
	column_quantity := len(m[0])

	result := make([][]float64, row_quantity)

	for i := 0; i < row_quantity; i++ {
		row := make([]float64, column_quantity)

		for j := 0; j < column_quantity; j++ {
			row[j] = m[i][j] * scalar
		}

		result[i] = row
	}

	return result, nil
}