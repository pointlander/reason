// Copyright 2024 The Reason Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/alixaxel/pagerank"
	"github.com/konimarti/kalman"
	"github.com/konimarti/lti"
	"github.com/pointlander/gradient/tf64"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
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

// Add adds two float32 matrices
func (m Matrix) Add(n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
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

// T tramsposes a matrix
func (m Matrix) T() Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func softmax(values []float64) {
	max := float64(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float64(0.0)
	for j, value := range values {
		values[j] = float64(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the page rank of x, y, z
func SelfAttention(x, y, z Matrix) (Matrix, []float64) {
	o := Matrix{
		Cols: z.Cols,
		Rows: y.Rows,
		Data: make([]float64, 0, z.Rows*y.Rows),
	}
	outputs, values := make([]float64, z.Cols), make([]float64, x.Rows)
	z = z.T()
	graph := pagerank.NewGraph()
	for i := 0; i < y.Rows; i++ {
		yy := y.Data[i*y.Cols : (i+1)*y.Cols]
		aa := 0.0
		for _, v := range yy {
			aa += v * v
		}
		aa = math.Sqrt(float64(aa))
		for j := 0; j < x.Rows; j++ {
			xx := x.Data[j*x.Cols : (j+1)*x.Cols]
			bb := 0.0
			for _, v := range xx {
				bb += v * v
			}
			bb = math.Sqrt(bb)
			values[j] = Dot(yy, xx)
			d := math.Abs(values[j] / (aa * bb))
			graph.Link(uint32(i), uint32(j), d)
		}
		softmax(values)

		for j := 0; j < z.Rows; j++ {
			zz := z.Data[j*z.Cols : (j+1)*z.Cols]
			outputs[j] = Dot(values, zz)
		}
		o.Data = append(o.Data, outputs...)
	}
	ranks := make([]float64, y.Rows)
	graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
		ranks[node] = rank
	})
	return o, ranks
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

const (
	iterations = 1024
)

// Sample samples the neural network
func Sample(stddev []float64, value Fisher, others *tf64.Set, l2 tf64.Meta, rng *rand.Rand) Matrix {
	samples := NewMatrix(3, iterations)
	for j := 0; j < iterations; j++ {
		transform := NewMatrix(4, 4)
		for k := 0; k < 4; k++ {
			sum := 1.0
			s := make([]float64, 3)
			for l := range s {
				v := rng.NormFloat64() * stddev[l] / 2
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
			samples.Data = append(samples.Data, a.X...)
			return true
		})
	}
	return samples
}

//go:embed p133.csv
var rose embed.FS

func load() [][]float64 {
	file, err := rose.Open("p133.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	r := csv.NewReader(file)
	r.Comma = ','

	lines, err := r.ReadAll()
	if err != nil {
		panic("could read data from file")
	}

	y := make([][]float64, len(lines))
	for row, line := range lines {
		tmp := make([]float64, len(line))
		for i, entry := range line {
			var value float64
			if value, err = strconv.ParseFloat(entry, 64); err != nil {
				fmt.Println("error parsing", line, i, entry)
				panic("could not parse all values")
			}
			tmp[i] = value
		}
		y[row] = tmp
	}
	return y
}

var colors = [...]color.RGBA{
	{R: 0xff, G: 0x00, B: 0x00, A: 255},
	{R: 0x00, G: 0xff, B: 0x00, A: 255},
	{R: 0x00, G: 0x00, B: 0xff, A: 255},
}

// Kalman is the kalman mode
func Kalman() {
	// load data
	y := load()[0]

	// define LTI system
	lti := lti.Discrete{
		Ad: mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		Bd: mat.NewDense(2, 2, nil),
		C:  mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		D:  mat.NewDense(2, 2, nil),
	}

	// system noise / process model covariance matrix ("Systemrauschen")
	Gd := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	ctx := kalman.Context{
		// initial state
		X: mat.NewVecDense(2, []float64{y[0], 2 * y[0]}),
		// initial covariance matrix
		P: mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
	}

	// create ROSE filter
	gammaR := 9.0
	alphaR := 0.5
	alphaM := 0.3
	filter := kalman.NewRoseFilter(lti, Gd, gammaR, alphaR, alphaM)

	// no control
	u := mat.NewVecDense(2, nil)

	points := make(plotter.XYs, 0, 8)
	points1 := make(plotter.XYs, 0, 8)
	for i, row := range y {
		// new measurement
		y := mat.NewVecDense(2, []float64{row, y[len(y)-1-i]})

		// apply filter
		filter.Apply(&ctx, y, u)

		// get corrected state vector
		state := filter.State()

		// print out input and output signals
		//fmt.Fprintf(file, "%3.8f,%3.8f\n", y.AtVec(0), state.AtVec(0))
		points = append(points, plotter.XY{X: float64(i), Y: float64(y.AtVec(0))})
		points1 = append(points1, plotter.XY{X: float64(i), Y: float64(state.AtVec(0))})
	}

	p := plot.New()

	p.Title.Text = "iteration vs value"
	p.X.Label.Text = "iteration"
	p.Y.Label.Text = "value"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = colors[0]
	p.Add(scatter)

	scatter, err = plotter.NewScatter(points1)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = colors[1]
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "values.png")
	if err != nil {
		panic(err)
	}
}

var (
	// FlagKalman is the kalman filter mode
	FlagKalman = flag.Bool("kalman", false, "kalman mode")
)

func main() {
	flag.Parse()

	if *FlagKalman {
		Kalman()
		return
	}

	rng := rand.New(rand.NewSource(1))
	iris := Load()
	stddev := make([]float64, 4)
	{
		min, max := math.MaxFloat64, 0.0
		for _, value := range iris {
			for _, v := range value.Measures {
				if v < min {
					min = v
				}
				if v > max {
					max = v
				}
			}
		}
		sum := make([]float64, 4)
		scale := max - min
		for _, value := range iris {
			for i, v := range value.Measures {
				vv := (v - min) / scale
				value.Measures[i] = vv
				sum[i] += vv
			}
		}
		for i := range sum {
			sum[i] /= float64(len(iris))
		}
		for _, value := range iris {
			for i, v := range value.Measures {
				diff := v - sum[i]
				stddev[i] += diff * diff
			}
		}
		for i := range stddev {
			stddev[i] /= float64(len(iris))
			stddev[i] = math.Sqrt(stddev[i])
		}
	}
	fmt.Println("stddev", stddev)
	set := tf64.NewSet()
	set.Add("w1", 4, 8)
	set.Add("b1", 8)
	set.Add("w2", 8, 3)
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

	correct = 0
	correct1 := 0
	correct2 := 0
	correct3 := 0
	for _, value := range iris {
		samples := Sample(stddev, value, &others, l2, rng)
		samples1 := Sample(stddev, value, &others, l2, rng)
		samples2 := Sample(stddev, value, &others, l2, rng)

		average := make([]float64, samples.Cols)
		for j := 0; j < iterations; j++ {
			for k := 0; k < samples.Cols; k++ {
				average[k] += samples.Data[j*samples.Cols+k]
			}
		}
		for j := range average {
			average[j] /= iterations
		}
		variance := make([]float64, samples.Cols)
		for j := 0; j < iterations; j++ {
			for k := 0; k < samples.Cols; k++ {
				diff := average[k] - samples.Data[j*samples.Cols+k]
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

		sa, ranks := SelfAttention(samples, samples1, samples2)
		sums := make([]float64, sa.Cols)
		for j := 0; j < sa.Rows; j++ {
			for k := 0; k < sa.Cols; k++ {
				sums[k] += ranks[j] * sa.Data[j*sa.Cols+k]
			}
		}
		{
			max, index := 0.0, 0
			for i, v := range sums {
				if v > max {
					max, index = v, i
				}
			}
			if index == Labels[value.Label] {
				correct3++
			}
		}
		votes := make([]int, sa.Cols)
		for j := 0; j < sa.Rows; j++ {
			max, index := 0.0, 0
			for k := 0; k < sa.Cols; k++ {
				v := sa.Data[j*sa.Cols+k]
				if v > max {
					max, index = v, k
				}
			}
			votes[index]++
		}
		max, index := 0, 0
		for i, v := range votes {
			if v > max {
				max, index = v, i
			}
		}
		if index == Labels[value.Label] {
			correct1++
		}

		average1 := make([]float64, sa.Cols)
		for j := 0; j < iterations; j++ {
			for k := 0; k < sa.Cols; k++ {
				average1[k] += sa.Data[j*sa.Cols+k]
			}
		}
		for j := range average1 {
			average1[j] /= iterations
		}
		variance1 := make([]float64, sa.Cols)
		for j := 0; j < iterations; j++ {
			for k := 0; k < sa.Cols; k++ {
				diff := average1[k] - sa.Data[j*sa.Cols+k]
				variance1[k] += diff * diff
			}
		}
		for j := range variance1 {
			variance1[j] /= iterations
			variance1[j] = math.Sqrt(variance1[j])
		}

		{
			min, index := math.MaxFloat64, 0
			for j, v := range variance1 {
				if v < min {
					min, index = v, j
				}
			}
			if index == Labels[value.Label] {
				correct2++
			}
		}
	}
	fmt.Println("variance correct", correct, float64(correct)/float64(len(iris)))
	fmt.Println("self attention correct", correct1, float64(correct1)/float64(len(iris)))
	fmt.Println("self attention variance correct", correct2, float64(correct2)/float64(len(iris)))
	fmt.Println("self attention page rank correct", correct3, float64(correct3)/float64(len(iris)))
}
