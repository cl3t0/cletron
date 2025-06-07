package utils

func CloneMatrix(m [][]float64) [][]float64 {
	newMatrix := make([][]float64, len(m))

	for i := range m {
		newMatrix[i] = make([]float64, len(m[i]))
		copy(newMatrix[i], m[i])
	}

	return newMatrix
}
