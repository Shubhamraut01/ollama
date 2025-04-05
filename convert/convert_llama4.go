package convert

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type llama4Model struct {
	ModelParameters
	TextModel struct {
		llamaModel
		NumExpertsPerToken     uint32 `json:"num_experts_per_tok"`
		NumLocalExperts        uint32 `json:"num_local_experts"`
		InterleaveMOELayerStep uint32 `json:"interleave_moe_layer_step"`
	} `json:"text_config"`
	VisionModel struct {
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		ImageSize         uint32  `json:"image_size"`
		PatchSize         uint32  `json:"patch_size"`
		RopeTheta         float32 `json:"rope_theta"`
		NormEpsilon       float32 `json:"norm_eps"`
		PixelShuffleRatio float32 `json:"pixel_shuffle_ratio"`
	} `json:"vision_config"`
}

// KV implements ModelConverter.
func (p *llama4Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "llama4"

	for k, v := range p.TextModel.KV(t) {
		if strings.HasPrefix(k, "llama.") {
			kv[strings.ReplaceAll(k, "llama.", "llama4.")] = v
		}
	}

	kv["llama4.expert_count"] = p.TextModel.NumExpertsPerToken
	kv["llama4.expert_used_count"] = p.TextModel.NumLocalExperts
	kv["llama4.interleave_moe_layer_step"] = p.TextModel.InterleaveMOELayerStep

	kv["llama4.vision.block_count"] = p.VisionModel.NumHiddenLayers
	kv["llama4.vision.embedding_length"] = p.VisionModel.IntermediateSize
	kv["llama4.vision.attention.head_count"] = p.VisionModel.NumAttentionHeads
	kv["llama4.vision.image_size"] = p.VisionModel.ImageSize
	kv["llama4.vision.patch_size"] = p.VisionModel.PatchSize
	kv["llama4.vision.rope.freq_base"] = p.VisionModel.RopeTheta
	kv["llama4.vision.layer_norm_epsilon"] = p.VisionModel.NormEpsilon
	kv["llama4.vision.pixel_shuffle_ratio"] = p.VisionModel.PixelShuffleRatio
	return kv
}

// Replacements implements ModelConverter.
func (p *llama4Model) Replacements() []string {
	return append(
		p.TextModel.Replacements(),
		"language_model", "",
		"vision_model", "v",
	)
}

// Tensors implements ModelConverter.
func (p *llama4Model) Tensors(ts []Tensor) []ggml.Tensor {
	var visionTensors []Tensor
	textTensors := slices.DeleteFunc(ts, func(t Tensor) bool {
		if strings.HasPrefix(t.Name(), "vision_model") {
			visionTensors = append(visionTensors, t)
			return true
		}

		return false
	})

	out := p.TextModel.Tensors(textTensors)
	for _, t := range visionTensors {
		out = append(out, ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

var _ ModelConverter = (*llama4Model)(nil)
