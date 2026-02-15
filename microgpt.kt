/**
 * The most atomic way to train and inference a GPT in pure, dependency-free Kotlin.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * @karpathy (original Python), converted to Kotlin
 */

import java.io.File
import kotlin.math.*

// --- RNG (xoshiro256** seeded with 42) ---
class Rng(seed: Long) {
    private var s0: Long
    private var s1: Long
    private var s2: Long
    private var s3: Long

    init {
        var sm = seed
        fun smNext(): Long {
            sm += -7046029254386353131L // 0x9e3779b97f4a7c15 as signed
            var z = sm
            z = (z xor (z ushr 30)) * -4658895280553007687L
            z = (z xor (z ushr 27)) * -7723592293110705685L
            return z xor (z ushr 31)
        }
        s0 = smNext(); s1 = smNext(); s2 = smNext(); s3 = smNext()
    }

    private fun rotl(x: Long, k: Int): Long = (x shl k) or (x ushr (64 - k))

    fun nextU64(): Long {
        val result = rotl(s1 * 5, 7) * 9
        val t = s1 shl 17
        s2 = s2 xor s0
        s3 = s3 xor s1
        s1 = s1 xor s2
        s0 = s0 xor s3
        s2 = s2 xor t
        s3 = rotl(s3, 45)
        return result
    }

    fun nextF64(): Double = (nextU64() ushr 11).toDouble() / (1L shl 53).toDouble()

    fun gauss(mean: Double, std: Double): Double {
        val u1 = maxOf(nextF64(), 1e-30)
        val u2 = nextF64()
        return mean + std * sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    fun <T> shuffle(list: MutableList<T>) {
        for (i in list.size - 1 downTo 1) {
            val j = ((nextU64() ushr 1) % (i + 1)).toInt() // ushr 1 to avoid negative modulo
            val tmp = list[i]; list[i] = list[j]; list[j] = tmp
        }
    }

    fun weightedChoice(weights: List<Double>): Int {
        val total = weights.sum()
        var r = nextF64() * total
        for ((i, w) in weights.withIndex()) {
            r -= w
            if (r <= 0) return i
        }
        return weights.size - 1
    }
}

// --- Let there be Autograd ---
class Value(
    var data: Double,
    val children: List<Value> = emptyList(),
    val localGrads: List<Double> = emptyList()
) {
    var grad: Double = 0.0

    operator fun plus(other: Value) =
        Value(data + other.data, listOf(this, other), listOf(1.0, 1.0))
    fun plusScalar(s: Double) = this + Value(s)

    operator fun times(other: Value) =
        Value(data * other.data, listOf(this, other), listOf(other.data, data))
    fun timesScalar(s: Double) = this * Value(s)

    fun vpow(exp: Double) =
        Value(data.pow(exp), listOf(this), listOf(exp * data.pow(exp - 1)))

    fun vlog() = Value(ln(data), listOf(this), listOf(1.0 / data))

    fun vexp(): Value {
        val e = exp(data)
        return Value(e, listOf(this), listOf(e))
    }

    fun relu() = Value(maxOf(0.0, data), listOf(this), listOf(if (data > 0) 1.0 else 0.0))

    operator fun unaryMinus() = timesScalar(-1.0)
    operator fun minus(other: Value) = this + (-other)
    operator fun div(other: Value) = this * other.vpow(-1.0)
    fun divScalar(s: Double) = timesScalar(1.0 / s)

    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (v in visited) return
            visited.add(v)
            for (child in v.children) buildTopo(child)
            topo.add(v)
        }
        buildTopo(this)
        grad = 1.0
        for (v in topo.reversed()) {
            for ((child, lg) in v.children.zip(v.localGrads)) {
                child.grad += lg * v.grad
            }
        }
    }
}

// --- Utility ---
fun sumVals(v: List<Value>): Value {
    var s = Value(0.0)
    for (x in v) s = s + x
    return s
}

// --- Model functions ---
fun linear(x: List<Value>, w: List<List<Value>>): List<Value> =
    w.map { wo ->
        var s = Value(0.0)
        for ((wi, xi) in wo.zip(x)) s = s + wi * xi
        s
    }

fun softmax(logits: List<Value>): List<Value> {
    val maxVal = logits.maxOf { it.data }
    val exps = logits.map { (it - Value(maxVal)).vexp() }
    val total = sumVals(exps)
    return exps.map { it / total }
}

fun rmsnorm(x: List<Value>): List<Value> {
    var ms = Value(0.0)
    for (xi in x) ms = ms + xi * xi
    ms = ms.divScalar(x.size.toDouble())
    val scale = ms.plusScalar(1e-5).vpow(-0.5)
    return x.map { it * scale }
}

// --- Main ---
fun main() {
    val rng = Rng(42)

    // Let there be an input dataset
    val file = File("input.txt")
    if (!file.exists()) {
        println("Error: input.txt not found. Please download it first.")
        return
    }
    val docs = file.readText().trim().split("\n")
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .toMutableList()
    rng.shuffle(docs)
    println("num docs: ${docs.size}")

    // Let there be a Tokenizer
    val uchars = docs.joinToString("").toSortedSet().toList()
    val bos = uchars.size
    val vocabSize = uchars.size + 1
    println("vocab size: $vocabSize")

    fun charIndex(c: Char): Int = uchars.indexOf(c)

    // Initialize the parameters
    val nEmbd = 16
    val nHead = 4
    val nLayer = 1
    val blockSize = 16
    val headDim = nEmbd / nHead

    fun makeMatrix(nout: Int, nin: Int): List<List<Value>> =
        List(nout) { List(nin) { Value(rng.gauss(0.0, 0.08)) } }

    val wte = makeMatrix(vocabSize, nEmbd)
    val wpe = makeMatrix(blockSize, nEmbd)
    val lmHead = makeMatrix(vocabSize, nEmbd)

    data class LayerWeights(
        val attnWq: List<List<Value>>,
        val attnWk: List<List<Value>>,
        val attnWv: List<List<Value>>,
        val attnWo: List<List<Value>>,
        val mlpFc1: List<List<Value>>,
        val mlpFc2: List<List<Value>>,
    )

    val layers = List(nLayer) {
        LayerWeights(
            attnWq = makeMatrix(nEmbd, nEmbd),
            attnWk = makeMatrix(nEmbd, nEmbd),
            attnWv = makeMatrix(nEmbd, nEmbd),
            attnWo = makeMatrix(nEmbd, nEmbd),
            mlpFc1 = makeMatrix(4 * nEmbd, nEmbd),
            mlpFc2 = makeMatrix(nEmbd, 4 * nEmbd),
        )
    }

    // Flatten params
    val params = mutableListOf<Value>()
    for (row in wte) for (p in row) params.add(p)
    for (row in wpe) for (p in row) params.add(p)
    for (row in lmHead) for (p in row) params.add(p)
    for (lw in layers) {
        for (mat in listOf(lw.attnWq, lw.attnWk, lw.attnWv, lw.attnWo, lw.mlpFc1, lw.mlpFc2)) {
            for (row in mat) for (p in row) params.add(p)
        }
    }
    println("num params: ${params.size}")

    // GPT function
    fun gpt(tokenId: Int, posId: Int,
            keys: MutableList<MutableList<List<Value>>>,
            values: MutableList<MutableList<List<Value>>>): List<Value> {
        val tokEmb = wte[tokenId]
        val posEmb = wpe[posId]
        var x = List(nEmbd) { i -> tokEmb[i] + posEmb[i] }
        x = rmsnorm(x)

        for (li in 0 until nLayer) {
            // 1) Multi-head attention block
            val xResidual = x.toList()
            x = rmsnorm(x)
            val q = linear(x, layers[li].attnWq)
            val k = linear(x, layers[li].attnWk)
            val v = linear(x, layers[li].attnWv)
            keys[li].add(k)
            values[li].add(v)
            val xAttn = mutableListOf<Value>()
            for (h in 0 until nHead) {
                val hs = h * headDim
                val qH = q.subList(hs, hs + headDim)
                val kH = keys[li].map { ki -> ki.subList(hs, hs + headDim) }
                val vH = values[li].map { vi -> vi.subList(hs, hs + headDim) }
                val scale = sqrt(headDim.toDouble())
                val attnLogits = kH.map { kt ->
                    var dot = Value(0.0)
                    for (j in 0 until headDim) dot = dot + qH[j] * kt[j]
                    dot.divScalar(scale)
                }
                val attnWeights = softmax(attnLogits)
                for (j in 0 until headDim) {
                    var s = Value(0.0)
                    for (t in vH.indices) s = s + attnWeights[t] * vH[t][j]
                    xAttn.add(s)
                }
            }
            x = linear(xAttn, layers[li].attnWo)
            x = List(nEmbd) { i -> x[i] + xResidual[i] }
            // 2) MLP block
            val xResidual2 = x.toList()
            x = rmsnorm(x)
            x = linear(x, layers[li].mlpFc1)
            x = x.map { it.relu() }
            x = linear(x, layers[li].mlpFc2)
            x = List(nEmbd) { i -> x[i] + xResidual2[i] }
        }

        return linear(x, lmHead)
    }

    // Let there be Adam
    val learningRate = 0.01
    val beta1 = 0.85
    val beta2 = 0.99
    val epsAdam = 1e-8
    val mBuf = DoubleArray(params.size)
    val vBuf = DoubleArray(params.size)

    // Repeat in sequence
    val numSteps = 1000
    for (step in 0 until numSteps) {
        // Take single document, tokenize it
        val doc = docs[step % docs.size]
        val tokens = mutableListOf(bos)
        for (c in doc) tokens.add(charIndex(c))
        tokens.add(bos)
        val n = minOf(blockSize, tokens.size - 1)

        // Forward
        val keys = MutableList(nLayer) { mutableListOf<List<Value>>() }
        val vals = MutableList(nLayer) { mutableListOf<List<Value>>() }
        val losses = mutableListOf<Value>()
        for (posId in 0 until n) {
            val tokenId = tokens[posId]
            val targetId = tokens[posId + 1]
            val logits = gpt(tokenId, posId, keys, vals)
            val probs = softmax(logits)
            val lossT = -(probs[targetId].vlog())
            losses.add(lossT)
        }
        val loss = sumVals(losses).divScalar(n.toDouble())

        // Backward
        loss.backward()

        // Adam optimizer update
        val lrT = learningRate * (1.0 - step.toDouble() / numSteps)
        for ((i, p) in params.withIndex()) {
            mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad
            vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad
            val mHat = mBuf[i] / (1.0 - beta1.pow(step + 1))
            val vHat = vBuf[i] / (1.0 - beta2.pow(step + 1))
            p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
            p.grad = 0.0
        }

        println("step %4d / %4d | loss %.4f".format(step + 1, numSteps, loss.data))
    }

    // Inference
    val temperature = 0.5
    println("\n--- inference (new, hallucinated names) ---")
    for (sampleIdx in 0 until 20) {
        val keys = MutableList(nLayer) { mutableListOf<List<Value>>() }
        val vals = MutableList(nLayer) { mutableListOf<List<Value>>() }
        var tokenId = bos
        val sample = mutableListOf<Char>()
        for (posId in 0 until blockSize) {
            val logits = gpt(tokenId, posId, keys, vals)
            val tempLogits = logits.map { it.divScalar(temperature) }
            val probs = softmax(tempLogits)
            val weights = probs.map { it.data }
            tokenId = rng.weightedChoice(weights)
            if (tokenId == bos) break
            sample.add(uchars[tokenId])
        }
        println("sample %2d: %s".format(sampleIdx + 1, sample.joinToString("")))
    }
}
