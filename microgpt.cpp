/**
 * The most atomic way to train and inference a GPT in pure, dependency-free C++.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * @karpathy (original Python), converted to C++
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <memory>
#include <functional>
#include <numeric>

// Let there be order among chaos
static std::mt19937 rng(42);

// --- Let there be Autograd ---
struct Value : std::enable_shared_from_this<Value> {
    double data;                                         // scalar value of this node calculated during forward pass
    double grad;                                         // derivative of the loss w.r.t. this node
    std::vector<std::shared_ptr<Value>> _children;       // children of this node in the computation graph
    std::vector<double> _local_grads;                    // local derivative of this node w.r.t. its children

    Value(double data, std::vector<std::shared_ptr<Value>> children = {},
          std::vector<double> local_grads = {})
        : data(data), grad(0.0), _children(std::move(children)), _local_grads(std::move(local_grads)) {}
};

using Val = std::shared_ptr<Value>;

Val val(double d) { return std::make_shared<Value>(d); }

Val operator+(const Val& a, const Val& b) {
    return std::make_shared<Value>(a->data + b->data, std::vector<Val>{a, b}, std::vector<double>{1.0, 1.0});
}
Val operator+(const Val& a, double b) { return a + val(b); }
Val operator+(double a, const Val& b) { return val(a) + b; }

Val operator*(const Val& a, const Val& b) {
    return std::make_shared<Value>(a->data * b->data, std::vector<Val>{a, b}, std::vector<double>{b->data, a->data});
}
Val operator*(const Val& a, double b) { return a * val(b); }
Val operator*(double a, const Val& b) { return val(a) * b; }

Val vpow(const Val& a, double b) {
    return std::make_shared<Value>(std::pow(a->data, b), std::vector<Val>{a}, std::vector<double>{b * std::pow(a->data, b - 1)});
}

Val vlog(const Val& a) {
    return std::make_shared<Value>(std::log(a->data), std::vector<Val>{a}, std::vector<double>{1.0 / a->data});
}

Val vexp(const Val& a) {
    double e = std::exp(a->data);
    return std::make_shared<Value>(e, std::vector<Val>{a}, std::vector<double>{e});
}

Val vrelu(const Val& a) {
    return std::make_shared<Value>(std::max(0.0, a->data), std::vector<Val>{a}, std::vector<double>{a->data > 0 ? 1.0 : 0.0});
}

Val operator-(const Val& a) { return a * (-1.0); }
Val operator-(const Val& a, const Val& b) { return a + (-b); }
Val operator-(const Val& a, double b) { return a + (-b); }
Val operator-(double a, const Val& b) { return val(a) + (-b); }
Val operator/(const Val& a, const Val& b) { return a * vpow(b, -1.0); }
Val operator/(const Val& a, double b) { return a * val(1.0 / b); }
Val operator/(double a, const Val& b) { return val(a) * vpow(b, -1.0); }

void backward(const Val& root) {
    std::vector<Val> topo;
    std::set<Value*> visited;
    std::function<void(const Val&)> build_topo = [&](const Val& v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto& child : v->_children) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    build_topo(root);
    root->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto& v = *it;
        for (size_t i = 0; i < v->_children.size(); ++i) {
            v->_children[i]->grad += v->_local_grads[i] * v->grad;
        }
    }
}

// --- Utility types ---
using Vec = std::vector<Val>;
using Mat = std::vector<Vec>;

// --- Helper functions ---
Val sum(const Vec& v) {
    Val s = val(0.0);
    for (auto& x : v) s = s + x;
    return s;
}

// --- Model functions ---
Vec linear(const Vec& x, const Mat& w) {
    Vec out;
    out.reserve(w.size());
    for (auto& wo : w) {
        Val s = val(0.0);
        for (size_t i = 0; i < wo.size(); ++i) {
            s = s + wo[i] * x[i];
        }
        out.push_back(s);
    }
    return out;
}

Vec softmax(const Vec& logits) {
    double max_val = -1e30;
    for (auto& v : logits) if (v->data > max_val) max_val = v->data;
    Vec exps;
    exps.reserve(logits.size());
    for (auto& v : logits) exps.push_back(vexp(v - max_val));
    Val total = sum(exps);
    Vec out;
    out.reserve(exps.size());
    for (auto& e : exps) out.push_back(e / total);
    return out;
}

Vec rmsnorm(const Vec& x) {
    Val ms = val(0.0);
    for (auto& xi : x) ms = ms + xi * xi;
    ms = ms / (double)x.size();
    Val scale = vpow(ms + 1e-5, -0.5);
    Vec out;
    out.reserve(x.size());
    for (auto& xi : x) out.push_back(xi * scale);
    return out;
}

// --- Main ---
int main() {
    // Let there be an input dataset `docs`
    std::ifstream file("input.txt");
    if (!file.is_open()) {
        std::cerr << "Error: input.txt not found. Please download it first." << std::endl;
        return 1;
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::vector<std::string> docs;
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        // trim
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            docs.push_back(line.substr(start, end - start + 1));
        }
    }
    std::shuffle(docs.begin(), docs.end(), rng);
    std::cout << "num docs: " << docs.size() << std::endl;

    // Let there be a Tokenizer
    std::set<char> char_set;
    for (auto& d : docs) for (char c : d) char_set.insert(c);
    std::vector<char> uchars(char_set.begin(), char_set.end());
    std::sort(uchars.begin(), uchars.end());
    int BOS = (int)uchars.size();          // token id for Beginning of Sequence
    int vocab_size = (int)uchars.size() + 1; // +1 for BOS
    std::cout << "vocab size: " << vocab_size << std::endl;

    auto char_index = [&](char c) -> int {
        for (int i = 0; i < (int)uchars.size(); ++i) if (uchars[i] == c) return i;
        return -1;
    };

    // Initialize the parameters
    int n_embd = 16;
    int n_head = 4;
    int n_layer = 1;
    int block_size = 16;
    int head_dim = n_embd / n_head;

    std::normal_distribution<double> gauss(0.0, 0.08);
    auto matrix = [&](int nout, int nin) -> Mat {
        Mat m(nout, Vec(nin));
        for (int i = 0; i < nout; ++i)
            for (int j = 0; j < nin; ++j)
                m[i][j] = val(gauss(rng));
        return m;
    };

    // state_dict
    std::map<std::string, Mat> state_dict;
    state_dict["wte"] = matrix(vocab_size, n_embd);
    state_dict["wpe"] = matrix(block_size, n_embd);
    state_dict["lm_head"] = matrix(vocab_size, n_embd);
    for (int i = 0; i < n_layer; ++i) {
        std::string prefix = "layer" + std::to_string(i);
        state_dict[prefix + ".attn_wq"] = matrix(n_embd, n_embd);
        state_dict[prefix + ".attn_wk"] = matrix(n_embd, n_embd);
        state_dict[prefix + ".attn_wv"] = matrix(n_embd, n_embd);
        state_dict[prefix + ".attn_wo"] = matrix(n_embd, n_embd);
        state_dict[prefix + ".mlp_fc1"] = matrix(4 * n_embd, n_embd);
        state_dict[prefix + ".mlp_fc2"] = matrix(n_embd, 4 * n_embd);
    }

    // Flatten params
    std::vector<Val> params;
    for (auto& [name, mat] : state_dict) {
        for (auto& row : mat) {
            for (auto& p : row) {
                params.push_back(p);
            }
        }
    }
    std::cout << "num params: " << params.size() << std::endl;

    // GPT function
    auto gpt = [&](int token_id, int pos_id,
                    std::vector<std::vector<Vec>>& keys,
                    std::vector<std::vector<Vec>>& values) -> Vec {
        Vec tok_emb = state_dict["wte"][token_id];
        Vec pos_emb = state_dict["wpe"][pos_id];
        Vec x(n_embd);
        for (int i = 0; i < n_embd; ++i) x[i] = tok_emb[i] + pos_emb[i];
        x = rmsnorm(x);

        for (int li = 0; li < n_layer; ++li) {
            std::string prefix = "layer" + std::to_string(li);
            // 1) Multi-head attention block
            Vec x_residual = x;
            x = rmsnorm(x);
            Vec q = linear(x, state_dict[prefix + ".attn_wq"]);
            Vec k = linear(x, state_dict[prefix + ".attn_wk"]);
            Vec v = linear(x, state_dict[prefix + ".attn_wv"]);
            keys[li].push_back(k);
            values[li].push_back(v);
            Vec x_attn;
            for (int h = 0; h < n_head; ++h) {
                int hs = h * head_dim;
                Vec q_h(q.begin() + hs, q.begin() + hs + head_dim);
                std::vector<Vec> k_h, v_h;
                for (auto& ki : keys[li]) k_h.push_back(Vec(ki.begin() + hs, ki.begin() + hs + head_dim));
                for (auto& vi : values[li]) v_h.push_back(Vec(vi.begin() + hs, vi.begin() + hs + head_dim));
                Vec attn_logits;
                double scale = std::sqrt((double)head_dim);
                for (size_t t = 0; t < k_h.size(); ++t) {
                    Val dot = val(0.0);
                    for (int j = 0; j < head_dim; ++j) dot = dot + q_h[j] * k_h[t][j];
                    attn_logits.push_back(dot / scale);
                }
                Vec attn_weights = softmax(attn_logits);
                for (int j = 0; j < head_dim; ++j) {
                    Val s = val(0.0);
                    for (size_t t = 0; t < v_h.size(); ++t) s = s + attn_weights[t] * v_h[t][j];
                    x_attn.push_back(s);
                }
            }
            x = linear(x_attn, state_dict[prefix + ".attn_wo"]);
            for (int i = 0; i < n_embd; ++i) x[i] = x[i] + x_residual[i];
            // 2) MLP block
            x_residual = x;
            x = rmsnorm(x);
            x = linear(x, state_dict[prefix + ".mlp_fc1"]);
            for (auto& xi : x) xi = vrelu(xi);
            x = linear(x, state_dict[prefix + ".mlp_fc2"]);
            for (int i = 0; i < n_embd; ++i) x[i] = x[i] + x_residual[i];
        }

        return linear(x, state_dict["lm_head"]);
    };

    // Let there be Adam
    double learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<double> m_buf(params.size(), 0.0);
    std::vector<double> v_buf(params.size(), 0.0);

    // Repeat in sequence
    int num_steps = 1000;
    for (int step = 0; step < num_steps; ++step) {
        // Take single document, tokenize it
        std::string doc = docs[step % docs.size()];
        std::vector<int> tokens;
        tokens.push_back(BOS);
        for (char c : doc) tokens.push_back(char_index(c));
        tokens.push_back(BOS);
        int n = std::min(block_size, (int)tokens.size() - 1);

        // Forward
        std::vector<std::vector<Vec>> keys(n_layer), values(n_layer);
        Vec losses;
        for (int pos_id = 0; pos_id < n; ++pos_id) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            Vec logits = gpt(token_id, pos_id, keys, values);
            Vec probs = softmax(logits);
            Val loss_t = -vlog(probs[target_id]);
            losses.push_back(loss_t);
        }
        Val loss = (1.0 / n) * sum(losses);

        // Backward
        backward(loss);

        // Adam optimizer update
        double lr_t = learning_rate * (1.0 - (double)step / num_steps);
        for (size_t i = 0; i < params.size(); ++i) {
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * params[i]->grad;
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * params[i]->grad * params[i]->grad;
            double m_hat = m_buf[i] / (1.0 - std::pow(beta1, step + 1));
            double v_hat = v_buf[i] / (1.0 - std::pow(beta2, step + 1));
            params[i]->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps_adam);
            params[i]->grad = 0.0;
        }

        printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss->data);
    }

    // Inference
    double temperature = 0.5;
    std::cout << "\n--- inference (new, hallucinated names) ---" << std::endl;
    for (int sample_idx = 0; sample_idx < 20; ++sample_idx) {
        std::vector<std::vector<Vec>> keys(n_layer), values(n_layer);
        int token_id = BOS;
        std::string sample;
        for (int pos_id = 0; pos_id < block_size; ++pos_id) {
            Vec logits = gpt(token_id, pos_id, keys, values);
            Vec temp_logits;
            for (auto& l : logits) temp_logits.push_back(l / temperature);
            Vec probs = softmax(temp_logits);
            std::vector<double> weights;
            for (auto& p : probs) weights.push_back(p->data);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);
            if (token_id == BOS) break;
            sample += uchars[token_id];
        }
        printf("sample %2d: %s\n", sample_idx + 1, sample.c_str());
    }

    return 0;
}
