#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {

struct Profile {
    double vision_encode_ms;
    double prefill_ms;
    double decode_ms;
    double total_ms;
    uint32_t generated_tokens;
};

struct Handle {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_adapter_lora * lora = nullptr;
    mtmd_context * vision = nullptr;
    uint32_t n_batch = 0;

    ~Handle() {
        if (vision) mtmd_free(vision);
        if (context) llama_free(context);
        if (lora) llama_adapter_lora_free(lora);
        if (model) llama_model_free(model);
    }
};

std::once_flag backend_once;

void quiet_log(enum ggml_log_level, const char *, void *) {}

double elapsed_ms(const std::chrono::steady_clock::time_point & start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

void copy_error(char * destination, size_t capacity, const std::string & message) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", message.c_str());
}

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::string format_chat(llama_model * model, const std::string & content) {
    const char * template_name = llama_model_chat_template(model, nullptr);
    check(template_name != nullptr, "GGUF model has no chat template");

    llama_chat_message message = {"user", content.c_str()};
    int32_t needed = llama_chat_apply_template(
        template_name, &message, 1, true, nullptr, 0);
    check(needed > 0, "llama.cpp failed to size chat template");

    std::vector<char> buffer(static_cast<size_t>(needed) + 1, '\0');
    int32_t written = llama_chat_apply_template(
        template_name, &message, 1, true, buffer.data(),
        static_cast<int32_t>(buffer.size()));
    check(written >= 0 && written <= needed, "llama.cpp failed to apply chat template");
    return std::string(buffer.data(), static_cast<size_t>(written));
}

void eval_prompt(Handle & handle, const uint8_t * rgb, uint32_t width,
                 uint32_t height, const std::string & prompt, Profile & profile) {
    mtmd_bitmap * raw_bitmap = mtmd_bitmap_init(width, height, rgb);
    check(raw_bitmap != nullptr, "mtmd_bitmap_init failed");
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
        raw_bitmap, mtmd_bitmap_free);

    const std::string content = std::string(mtmd_default_marker()) + prompt;
    const std::string formatted = format_chat(handle.model, content);
    mtmd_input_text text = {
        formatted.data(), formatted.size(), true, true,
    };
    const mtmd_bitmap * bitmap_ptr = bitmap.get();

    mtmd_input_chunks * raw_chunks = mtmd_input_chunks_init();
    check(raw_chunks != nullptr, "mtmd_input_chunks_init failed");
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        raw_chunks, mtmd_input_chunks_free);

    check(mtmd_tokenize(handle.vision, chunks.get(), &text, &bitmap_ptr, 1) == 0,
          "mtmd_tokenize failed");
    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    check(n_chunks > 0, "mtmd_tokenize returned no chunks");

    for (size_t index = 0; index < n_chunks; ++index) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), index);
        check(chunk != nullptr, "mtmd_input_chunks_get failed");
        const auto kind = mtmd_input_chunk_get_type(chunk);
        llama_pos new_n_past = n_past;
        if (kind == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            const auto start = std::chrono::steady_clock::now();
            check(mtmd_helper_eval_chunk_single(
                      handle.vision, handle.context, chunk, n_past, 0,
                      static_cast<int32_t>(handle.n_batch), index + 1 == n_chunks,
                      &new_n_past) == 0,
                  "mtmd_helper_eval_chunk_single failed");
            profile.prefill_ms += elapsed_ms(start);
        } else {
            const auto encode_start = std::chrono::steady_clock::now();
            check(mtmd_encode_chunk(handle.vision, chunk) == 0,
                  "mtmd_encode_chunk failed");
            profile.vision_encode_ms += elapsed_ms(encode_start);
            float * embeddings = mtmd_get_output_embd(handle.vision);
            check(embeddings != nullptr, "mtmd_get_output_embd returned null");

            const auto decode_start = std::chrono::steady_clock::now();
            check(mtmd_helper_decode_image_chunk(
                      handle.vision, handle.context, chunk, embeddings, n_past, 0,
                      static_cast<int32_t>(handle.n_batch), &new_n_past,
                      nullptr, nullptr) == 0,
                  "mtmd_helper_decode_image_chunk failed");
            profile.prefill_ms += elapsed_ms(decode_start);
        }
        n_past = new_n_past;
    }
}

std::string greedy_decode(Handle & handle, uint32_t max_tokens, Profile & profile) {
    const llama_vocab * vocab = llama_model_get_vocab(handle.model);
    check(vocab != nullptr, "llama_model_get_vocab failed");
    llama_sampler * sampler = llama_sampler_init_greedy();
    check(sampler != nullptr, "llama_sampler_init_greedy failed");
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_guard(
        sampler, llama_sampler_free);

    std::string output;
    const auto decode_start = std::chrono::steady_clock::now();
    for (uint32_t index = 0; index < max_tokens; ++index) {
        llama_token token = llama_sampler_sample(sampler, handle.context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        char piece[256];
        const int32_t n_piece = llama_token_to_piece(
            vocab, token, piece, sizeof(piece), 0, true);
        check(n_piece >= 0 && n_piece < static_cast<int32_t>(sizeof(piece)),
              "llama_token_to_piece failed or output was too long");
        output.append(piece, static_cast<size_t>(n_piece));

        llama_sampler_accept(sampler, token);
        const llama_batch batch = llama_batch_get_one(&token, 1);
        check(llama_decode(handle.context, batch) == 0, "llama_decode failed");
        ++profile.generated_tokens;
    }
    profile.decode_ms = elapsed_ms(decode_start);
    return output;
}

}  // namespace

extern "C" {

void * aerodpo_llamacpp_local_create(
    const char * text_path, const char * lora_path, const char * mmproj_path,
    uint32_t context_size, char * error, size_t error_capacity) {
    try {
        check(text_path && lora_path && mmproj_path, "model paths must not be null");
        check(context_size >= 256, "context must be at least 256");
        std::call_once(backend_once, [] {
            ggml_backend_load_all();
            llama_log_set(quiet_log, nullptr);
            mtmd_helper_log_set(quiet_log, nullptr);
        });

        auto handle = std::make_unique<Handle>();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;
        handle->model = llama_model_load_from_file(text_path, model_params);
        check(handle->model != nullptr, "failed to load text GGUF");

        handle->lora = llama_adapter_lora_init(handle->model, lora_path);
        check(handle->lora != nullptr, "failed to load language LoRA GGUF");

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = context_size;
        context_params.n_batch = context_size;
        context_params.n_ubatch = context_size;
        handle->context = llama_init_from_model(handle->model, context_params);
        check(handle->context != nullptr, "failed to create llama.cpp context");

        llama_adapter_lora * adapters[] = {handle->lora};
        float scales[] = {1.0f};
        check(llama_set_adapters_lora(handle->context, adapters, 1, scales) == 0,
              "failed to attach language LoRA");

        mtmd_context_params vision_params = mtmd_context_params_default();
        handle->vision = mtmd_init_from_file(mmproj_path, handle->model, vision_params);
        check(handle->vision != nullptr, "failed to load mmproj GGUF");
        handle->n_batch = context_params.n_batch;
        return handle.release();
    } catch (const std::exception & exception) {
        copy_error(error, error_capacity, exception.what());
        return nullptr;
    }
}

int aerodpo_llamacpp_local_generate(
    void * opaque_handle, const uint8_t * rgb, uint32_t width, uint32_t height,
    const char * prompt, uint32_t max_tokens, char * output,
    size_t output_capacity, Profile * profile, char * error, size_t error_capacity) {
    try {
        check(opaque_handle && rgb && prompt && output && profile,
              "handle, rgb, prompt, output, and profile must not be null");
        check(width == 384 && height == 768,
              "bridge requires RGB image dimensions 384x768");
        check(max_tokens > 0, "max_tokens must be positive");

        auto & handle = *static_cast<Handle *>(opaque_handle);
        *profile = {};
        const auto total_start = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(handle.context), true);
        eval_prompt(handle, rgb, width, height, prompt, *profile);
        const std::string text = greedy_decode(handle, max_tokens, *profile);
        check(text.size() + 1 <= output_capacity, "model output exceeds output buffer");
        std::memcpy(output, text.data(), text.size());
        output[text.size()] = '\0';
        profile->total_ms = elapsed_ms(total_start);
        return 0;
    } catch (const std::exception & exception) {
        copy_error(error, error_capacity, exception.what());
        return 1;
    }
}

void aerodpo_llamacpp_local_free(void * opaque_handle) {
    delete static_cast<Handle *>(opaque_handle);
}

}  // extern "C"
