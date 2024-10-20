// Copyright 2024 The Reason Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// Dot computes the dot product
func Dot(x, y []float64) (z float64) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, Dot(mm, nn))
		}
	}
	return o
}

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for i := range item[:4] {
					f, err := strconv.ParseFloat(item[i], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[i] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

func main() {
	rng := rand.New(rand.NewSource(1))
	iris := Load()

	set := tf64.NewSet()
	set.Add("w1", 4, 4)
	set.Add("b1", 4)
	set.Add("w2", 4, 3)
	set.Add("b2", 3)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	others.Add("input", 4)
	others.Add("output", 3)

	for i := range others.Weights {
		w := others.Weights[i]
		w.X = w.X[:cap(w.X)]
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
	l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")))
	loss := tf64.Quadratic(l2, others.Get("output"))

	points := make(plotter.XYs, 0, 8)
	for i := 0; i < 33*len(iris); i++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		others.Zero()
		index := rng.Intn(len(iris))
		input := others.ByName["input"].X
		for j := range input {
			input[j] = iris[index].Measures[j]
		}
		output := others.ByName["output"].X
		for j := range output {
			output[j] = 0
		}
		output[Labels[iris[index].Label]] = 1

		set.Zero()
		cost := tf64.Gradient(loss).X[0]

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		if norm > 1 {
			scaling := 1 / norm
			for _, w := range set.Weights {
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
		} else {
			for _, w := range set.Weights {
				for l, d := range w.D {
					g := d
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	correct := 0
	for _, value := range iris {
		input := others.ByName["input"].X
		for j := range input {
			input[j] = value.Measures[j]
		}
		l2(func(a *tf64.V) bool {
			max, index := 0.0, 0
			for j, v := range a.X {
				if v > max {
					max, index = v, j
				}
			}
			if index == Labels[value.Label] {
				correct++
			}
			return true
		})
	}
	fmt.Println("correct", correct, float64(correct)/float64(len(iris)))

	const (
		iterations = 1024
	)
	correct = 0
	for _, value := range iris {
		samples := make([]float64, 0, 8)
		for j := 0; j < iterations; j++ {
			transform := NewMatrix(4, 4)
			for k := 0; k < 4; k++ {
				sum := 1.0
				s := make([]float64, 3)
				for l := range s {
					v := rng.NormFloat64() / 8
					sum -= v
					s[l] = v
				}
				index := 0
				for l := 0; l < 4; l++ {
					if k == l {
						transform.Data = append(transform.Data, sum)
					} else {
						transform.Data = append(transform.Data, s[index])
						index++
					}
				}
			}
			in := NewMatrix(4, 1)
			for k := 0; k < 4; k++ {
				in.Data = append(in.Data, value.Measures[k])
			}
			out := transform.MulT(in)
			input := others.ByName["input"].X
			for j := range input {
				input[j] = out.Data[j]
			}
			l2(func(a *tf64.V) bool {
				samples = append(samples, a.X...)
				return true
			})
		}
		average := make([]float64, 3)
		for j := 0; j < iterations; j++ {
			for k := 0; k < 3; k++ {
				average[k] += samples[j*3+k]
			}
		}
		for j := range average {
			average[j] /= iterations
		}
		variance := make([]float64, 3)
		for j := 0; j < iterations; j++ {
			for k := 0; k < 3; k++ {
				diff := average[k] - samples[j*3+k]
				variance[k] += diff * diff
			}
		}
		for j := range variance {
			variance[j] /= iterations
			variance[j] = math.Sqrt(variance[j])
		}

		input := others.ByName["input"].X
		for j := range input {
			input[j] = value.Measures[j]
		}
		l2(func(a *tf64.V) bool {
			min, index := math.MaxFloat64, 0
			for j, v := range variance {
				v /= a.X[j]
				if v < min {
					min, index = v, j
				}
			}
			if index == Labels[value.Label] {
				correct++
			}
			return true
		})
	}
	fmt.Println("correct", correct, float64(correct)/float64(len(iris)))
}
