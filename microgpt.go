// The most atomic way to train and inference a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// @karpathy (original Python), converted to Go

package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

// --- RNG (xoshiro256** seeded with 42) ---
type Rng struct {
	s [4]uint64
}

func NewRng(seed uint64) *Rng {
	r := &Rng{}
	sm := seed
	smNext := func() uint64 {
		sm += 0x9e3779b97f4a7c15
		z := sm
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		return z ^ (z >> 31)
	}
	r.s[0] = smNext()
	r.s[1] = smNext()
	r.s[2] = smNext()
	r.s[3] = smNext()
	return r
}

func rotl(x uint64, k uint) uint64 {
	return (x << k) | (x >> (64 - k))
}

func (r *Rng) NextU64() uint64 {
	result := rotl(r.s[1]*5, 7) * 9
	t := r.s[1] << 17
	r.s[2] ^= r.s[0]
	r.s[3] ^= r.s[1]
	r.s[1] ^= r.s[2]
	r.s[0] ^= r.s[3]
	r.s[2] ^= t
	r.s[3] = rotl(r.s[3], 45)
	return result
}

func (r *Rng) NextF64() float64 {
	return float64(r.NextU64()>>11) / float64(uint64(1)<<53)
}

func (r *Rng) Gauss(mean, std float64) float64 {
	u1 := math.Max(r.NextF64(), 1e-30)
	u2 := r.NextF64()
	return mean + std*math.Sqrt(-2.0*math.Log(u1))*math.Cos(2.0*math.Pi*u2)
}

func (r *Rng) Shuffle(n int, swap func(i, j int)) {
	for i := n - 1; i > 0; i-- {
		j := int(r.NextU64() % uint64(i+1))
		swap(i, j)
	}
}

func (r *Rng) WeightedChoice(weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	rv := r.NextF64() * total
	for i, w := range weights {
		rv -= w
		if rv <= 0 {
			return i
		}
	}
	return len(weights) - 1
}

// --- Let there be Autograd ---
type Value struct {
	Data       float64
	Grad       float64
	children   []*Value
	localGrads []float64
}

func NewValue(data float64, children []*Value, localGrads []float64) *Value {
	return &Value{Data: data, Grad: 0, children: children, localGrads: localGrads}
}

func Val(d float64) *Value {
	return NewValue(d, nil, nil)
}

func Add(a, b *Value) *Value {
	return NewValue(a.Data+b.Data, []*Value{a, b}, []float64{1, 1})
}

func AddScalar(a *Value, s float64) *Value {
	return Add(a, Val(s))
}

func Mul(a, b *Value) *Value {
	return NewValue(a.Data*b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
}

func MulScalar(a *Value, s float64) *Value {
	return Mul(a, Val(s))
}

func Pow(a *Value, exp float64) *Value {
	return NewValue(math.Pow(a.Data, exp), []*Value{a}, []float64{exp * math.Pow(a.Data, exp-1)})
}

func Log(a *Value) *Value {
	return NewValue(math.Log(a.Data), []*Value{a}, []float64{1.0 / a.Data})
}

func Exp(a *Value) *Value {
	e := math.Exp(a.Data)
	return NewValue(e, []*Value{a}, []float64{e})
}

func Relu(a *Value) *Value {
	d := 0.0
	g := 0.0
	if a.Data > 0 {
		d = a.Data
		g = 1.0
	}
	return NewValue(d, []*Value{a}, []float64{g})
}

func Neg(a *Value) *Value    { return MulScalar(a, -1) }
func Sub(a, b *Value) *Value { return Add(a, Neg(b)) }
func Div(a, b *Value) *Value { return Mul(a, Pow(b, -1)) }
func DivScalar(a *Value, s float64) *Value {
	return MulScalar(a, 1.0/s)
}

func Backward(root *Value) {
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(root)
	root.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j, child := range v.children {
			child.Grad += v.localGrads[j] * v.Grad
		}
	}
}

// --- Utility ---
func SumVals(v []*Value) *Value {
	s := Val(0)
	for _, x := range v {
		s = Add(s, x)
	}
	return s
}

// --- Model functions ---
func Linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for i, wo := range w {
		s := Val(0)
		for j := range wo {
			s = Add(s, Mul(wo[j], x[j]))
		}
		out[i] = s
	}
	return out
}

func Softmax(logits []*Value) []*Value {
	maxVal := logits[0].Data
	for _, v := range logits[1:] {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}
	exps := make([]*Value, len(logits))
	for i, v := range logits {
		exps[i] = Exp(AddScalar(v, -maxVal))
	}
	total := SumVals(exps)
	out := make([]*Value, len(exps))
	for i, e := range exps {
		out[i] = Div(e, total)
	}
	return out
}

func Rmsnorm(x []*Value) []*Value {
	ms := Val(0)
	for _, xi := range x {
		ms = Add(ms, Mul(xi, xi))
	}
	ms = DivScalar(ms, float64(len(x)))
	scale := Pow(AddScalar(ms, 1e-5), -0.5)
	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = Mul(xi, scale)
	}
	return out
}

// --- Layer weights ---
type LayerWeights struct {
	attnWq, attnWk, attnWv, attnWo [][]*Value
	mlpFc1, mlpFc2                  [][]*Value
}

func main() {
	rng := NewRng(42)

	// Let there be an input dataset
	file, err := os.Open("input.txt")
	if err != nil {
		fmt.Println("Error: input.txt not found. Please download it first.")
		return
	}
	defer file.Close()

	var docs []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	fmt.Printf("num docs: %d\n", len(docs))

	// Let there be a Tokenizer
	charSet := make(map[rune]bool)
	for _, d := range docs {
		for _, c := range d {
			charSet[c] = true
		}
	}
	uchars := make([]rune, 0, len(charSet))
	for c := range charSet {
		uchars = append(uchars, c)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	bos := len(uchars)
	vocabSize := len(uchars) + 1
	fmt.Printf("vocab size: %d\n", vocabSize)

	charIndex := func(c rune) int {
		for i, u := range uchars {
			if u == c {
				return i
			}
		}
		return -1
	}

	// Initialize the parameters
	nEmbd := 16
	nHead := 4
	nLayer := 1
	blockSize := 16
	headDim := nEmbd / nHead

	makeMatrix := func(nout, nin int) [][]*Value {
		m := make([][]*Value, nout)
		for i := range m {
			m[i] = make([]*Value, nin)
			for j := range m[i] {
				m[i][j] = Val(rng.Gauss(0, 0.08))
			}
		}
		return m
	}

	wte := makeMatrix(vocabSize, nEmbd)
	wpe := makeMatrix(blockSize, nEmbd)
	lmHead := makeMatrix(vocabSize, nEmbd)

	layers := make([]LayerWeights, nLayer)
	for i := range layers {
		layers[i] = LayerWeights{
			attnWq: makeMatrix(nEmbd, nEmbd),
			attnWk: makeMatrix(nEmbd, nEmbd),
			attnWv: makeMatrix(nEmbd, nEmbd),
			attnWo: makeMatrix(nEmbd, nEmbd),
			mlpFc1: makeMatrix(4*nEmbd, nEmbd),
			mlpFc2: makeMatrix(nEmbd, 4*nEmbd),
		}
	}

	// Flatten params
	var params []*Value
	for _, row := range wte {
		params = append(params, row...)
	}
	for _, row := range wpe {
		params = append(params, row...)
	}
	for _, row := range lmHead {
		params = append(params, row...)
	}
	for _, lw := range layers {
		for _, mat := range [][][]*Value{lw.attnWq, lw.attnWk, lw.attnWv, lw.attnWo, lw.mlpFc1, lw.mlpFc2} {
			for _, row := range mat {
				params = append(params, row...)
			}
		}
	}
	fmt.Printf("num params: %d\n", len(params))

	// GPT function
	gpt := func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		tokEmb := wte[tokenID]
		posEmb := wpe[posID]
		x := make([]*Value, nEmbd)
		for i := range x {
			x[i] = Add(tokEmb[i], posEmb[i])
		}
		x = Rmsnorm(x)

		for li := 0; li < nLayer; li++ {
			// 1) Multi-head attention block
			xResidual := make([]*Value, nEmbd)
			copy(xResidual, x)
			x = Rmsnorm(x)
			q := Linear(x, layers[li].attnWq)
			k := Linear(x, layers[li].attnWk)
			v := Linear(x, layers[li].attnWv)
			keys[li] = append(keys[li], k)
			values[li] = append(values[li], v)
			xAttn := make([]*Value, 0, nEmbd)
			for h := 0; h < nHead; h++ {
				hs := h * headDim
				qH := q[hs : hs+headDim]
				kH := make([][]*Value, len(keys[li]))
				vH := make([][]*Value, len(values[li]))
				for t := range keys[li] {
					kH[t] = keys[li][t][hs : hs+headDim]
					vH[t] = values[li][t][hs : hs+headDim]
				}
				scale := math.Sqrt(float64(headDim))
				attnLogits := make([]*Value, len(kH))
				for t := range kH {
					dot := Val(0)
					for j := 0; j < headDim; j++ {
						dot = Add(dot, Mul(qH[j], kH[t][j]))
					}
					attnLogits[t] = DivScalar(dot, scale)
				}
				attnWeights := Softmax(attnLogits)
				for j := 0; j < headDim; j++ {
					s := Val(0)
					for t := range vH {
						s = Add(s, Mul(attnWeights[t], vH[t][j]))
					}
					xAttn = append(xAttn, s)
				}
			}
			x = Linear(xAttn, layers[li].attnWo)
			for i := 0; i < nEmbd; i++ {
				x[i] = Add(x[i], xResidual[i])
			}
			// 2) MLP block
			xResidual2 := make([]*Value, nEmbd)
			copy(xResidual2, x)
			x = Rmsnorm(x)
			x = Linear(x, layers[li].mlpFc1)
			for i := range x {
				x[i] = Relu(x[i])
			}
			x = Linear(x, layers[li].mlpFc2)
			for i := 0; i < nEmbd; i++ {
				x[i] = Add(x[i], xResidual2[i])
			}
		}

		return Linear(x, lmHead)
	}

	// Let there be Adam
	learningRate := 0.01
	beta1 := 0.85
	beta2 := 0.99
	epsAdam := 1e-8
	mBuf := make([]float64, len(params))
	vBuf := make([]float64, len(params))

	// Repeat in sequence
	numSteps := 1000
	for step := 0; step < numSteps; step++ {
		// Take single document, tokenize it
		doc := docs[step%len(docs)]
		tokens := []int{bos}
		for _, c := range doc {
			tokens = append(tokens, charIndex(c))
		}
		tokens = append(tokens, bos)
		n := blockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		// Forward
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := range keys {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}
		var losses []*Value
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := gpt(tokenID, posID, keys, vals)
			probs := Softmax(logits)
			lossT := Neg(Log(probs[targetID]))
			losses = append(losses, lossT)
		}
		loss := DivScalar(SumVals(losses), float64(n))

		// Backward
		Backward(loss)

		// Adam optimizer update
		lrT := learningRate * (1.0 - float64(step)/float64(numSteps))
		for i, p := range params {
			mBuf[i] = beta1*mBuf[i] + (1-beta1)*p.Grad
			vBuf[i] = beta2*vBuf[i] + (1-beta2)*p.Grad*p.Grad
			mHat := mBuf[i] / (1.0 - math.Pow(beta1, float64(step+1)))
			vHat := vBuf[i] / (1.0 - math.Pow(beta2, float64(step+1)))
			p.Data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.Grad = 0
		}

		fmt.Printf("step %4d / %4d | loss %.4f\n", step+1, numSteps, loss.Data)
	}

	// Inference
	temperature := 0.5
	fmt.Println("\n--- inference (new, hallucinated names) ---")
	for sampleIdx := 0; sampleIdx < 20; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		vals := make([][][]*Value, nLayer)
		for i := range keys {
			keys[i] = make([][]*Value, 0)
			vals[i] = make([][]*Value, 0)
		}
		tokenID := bos
		var sample []rune
		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(tokenID, posID, keys, vals)
			tempLogits := make([]*Value, len(logits))
			for i, l := range logits {
				tempLogits[i] = DivScalar(l, temperature)
			}
			probs := Softmax(tempLogits)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.Data
			}
			tokenID = rng.WeightedChoice(weights)
			if tokenID == bos {
				break
			}
			sample = append(sample, uchars[tokenID])
		}
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, string(sample))
	}
}
