#ifndef PIPELINE_NPY_H
#define PIPELINE_NPY_H

#include <cstdint>
#include <cstdio>

#include <stdexcept>
#include <string>

/// @brief Streams a float32 .npy array of shape (rows, cols) to disk
/// @details Rows are appended as they are produced, so a generation's samples
/// (~3.8 GB for 25k games) never have to sit in memory at once. The header
/// has a fixed 128-byte size and is rewritten with the final row count on
/// close(), so the file is valid .npy only after close().
class NpyWriter {
 public:
  NpyWriter(const std::string &path, int64_t cols) : cols_{cols} {
    file_ = std::fopen(path.c_str(), "wb");
    if (file_ == nullptr)
      throw std::runtime_error("cannot open " + path);
    writeHeader();
  }
  NpyWriter(const NpyWriter &) = delete;
  NpyWriter &operator=(const NpyWriter &) = delete;
  ~NpyWriter() {
    if (file_ != nullptr)
      close();
  }

  void append(const float *data, int64_t rows) {
    const size_t n = static_cast<size_t>(rows * cols_);
    if (std::fwrite(data, sizeof(float), n, file_) != n)
      throw std::runtime_error("short write");
    rows_ += rows;
  }

  void close() {
    std::fseek(file_, 0, SEEK_SET);
    writeHeader();
    std::fclose(file_);
    file_ = nullptr;
  }

  int64_t rows() const noexcept { return rows_; }

 private:
  void writeHeader() {
    // magic, version 1.0, uint16 header length, then the dict padded with
    // spaces and ended by a newline so that the data starts at byte 128
    constexpr size_t kTotal = 128;
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows_) + ", " + std::to_string(cols_) +
                       "), }";
    const size_t prefix = 10;
    if (prefix + dict.size() + 1 > kTotal)
      throw std::runtime_error("npy header too long");
    dict.append(kTotal - prefix - dict.size() - 1, ' ');
    dict.push_back('\n');
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const uint16_t len = static_cast<uint16_t>(dict.size());
    std::fwrite(magic, 1, 8, file_);
    std::fwrite(&len, 2, 1, file_);  // little-endian host
    std::fwrite(dict.data(), 1, dict.size(), file_);
  }

  std::FILE *file_{nullptr};
  int64_t cols_;
  int64_t rows_{0};
};

#endif
