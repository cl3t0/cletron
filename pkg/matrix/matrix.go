package main

import (
	"errors"
	"fmt"
)

func dot_product(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	if len(m1[0]) != len(m2) {
		return nil, errors.New("the number of columns in the first matrix must be equal to the number of rows in the second matrix")
	}

	result := [][]float64{}

	for i := 0; i < len(m1); i++ {
		vector := []float64{}

		for j := 0; j < len(m1); j++ {
			acc := 0.0
			
			for k := 0; k < len(m2); k++ {
				m1_element := m1[i][k]
				m2_element := m2[k][j]

				acc += m1_element * m2_element
			}

			vector = append(vector, acc)
		}

		result = append(result, vector)
	}

	return result, nil
}

func main() {
	mt1 := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	mt2 := [][]float64{
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {7, 8, 9},
	}	

	new_matrix, error := dot_product(mt1, mt2)
	if error != nil {
		fmt.Println("Erro: ", error)
		return
	}

	fmt.Println(new_matrix)
}