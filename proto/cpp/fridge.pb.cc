// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: fridge.proto

#include "fridge.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace AN {
PROTOBUF_CONSTEXPR ImageChunk::ImageChunk(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.buffer_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ImageChunkDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ImageChunkDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ImageChunkDefaultTypeInternal() {}
  union {
    ImageChunk _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ImageChunkDefaultTypeInternal _ImageChunk_default_instance_;
PROTOBUF_CONSTEXPR DetectResult::DetectResult(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.detection_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct DetectResultDefaultTypeInternal {
  PROTOBUF_CONSTEXPR DetectResultDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~DetectResultDefaultTypeInternal() {}
  union {
    DetectResult _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 DetectResultDefaultTypeInternal _DetectResult_default_instance_;
PROTOBUF_CONSTEXPR Detection::Detection(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.x1_)*/0
  , /*decltype(_impl_.y1_)*/0
  , /*decltype(_impl_.x2_)*/0
  , /*decltype(_impl_.y2_)*/0
  , /*decltype(_impl_.confidence_)*/0
  , /*decltype(_impl_.id_)*/0
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct DetectionDefaultTypeInternal {
  PROTOBUF_CONSTEXPR DetectionDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~DetectionDefaultTypeInternal() {}
  union {
    Detection _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 DetectionDefaultTypeInternal _Detection_default_instance_;
}  // namespace AN
static ::_pb::Metadata file_level_metadata_fridge_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_fridge_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_fridge_2eproto = nullptr;

const uint32_t TableStruct_fridge_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::AN::ImageChunk, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::AN::ImageChunk, _impl_.buffer_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::AN::DetectResult, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::AN::DetectResult, _impl_.detection_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.x1_),
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.y1_),
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.x2_),
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.y2_),
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.confidence_),
  PROTOBUF_FIELD_OFFSET(::AN::Detection, _impl_.id_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::AN::ImageChunk)},
  { 7, -1, -1, sizeof(::AN::DetectResult)},
  { 14, -1, -1, sizeof(::AN::Detection)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::AN::_ImageChunk_default_instance_._instance,
  &::AN::_DetectResult_default_instance_._instance,
  &::AN::_Detection_default_instance_._instance,
};

const char descriptor_table_protodef_fridge_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\014fridge.proto\022\002AN\"\034\n\nImageChunk\022\016\n\006buff"
  "er\030\001 \001(\014\"0\n\014DetectResult\022 \n\tdetection\030\001 "
  "\003(\0132\r.AN.Detection\"[\n\tDetection\022\n\n\002x1\030\001 "
  "\001(\002\022\n\n\002y1\030\002 \001(\002\022\n\n\002x2\030\003 \001(\002\022\n\n\002y2\030\004 \001(\002\022"
  "\022\n\nconfidence\030\005 \001(\002\022\n\n\002id\030\006 \001(\0052E\n\016Objec"
  "tDetector\0223\n\013DetectImage\022\016.AN.ImageChunk"
  "\032\020.AN.DetectResult\"\000(\001BG\n)com.szu.refrig"
  "erator.proto.ObjectDetectorB\023ObjectDetec"
  "torProtoP\001\242\002\002ANb\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_fridge_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_fridge_2eproto = {
    false, false, 343, descriptor_table_protodef_fridge_2eproto,
    "fridge.proto",
    &descriptor_table_fridge_2eproto_once, nullptr, 0, 3,
    schemas, file_default_instances, TableStruct_fridge_2eproto::offsets,
    file_level_metadata_fridge_2eproto, file_level_enum_descriptors_fridge_2eproto,
    file_level_service_descriptors_fridge_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_fridge_2eproto_getter() {
  return &descriptor_table_fridge_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_fridge_2eproto(&descriptor_table_fridge_2eproto);
namespace AN {

// ===================================================================

class ImageChunk::_Internal {
 public:
};

ImageChunk::ImageChunk(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:AN.ImageChunk)
}
ImageChunk::ImageChunk(const ImageChunk& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ImageChunk* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.buffer_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _impl_.buffer_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.buffer_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_buffer().empty()) {
    _this->_impl_.buffer_.Set(from._internal_buffer(), 
      _this->GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:AN.ImageChunk)
}

inline void ImageChunk::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.buffer_){}
    , /*decltype(_impl_._cached_size_)*/{}
  };
  _impl_.buffer_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.buffer_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

ImageChunk::~ImageChunk() {
  // @@protoc_insertion_point(destructor:AN.ImageChunk)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ImageChunk::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.buffer_.Destroy();
}

void ImageChunk::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ImageChunk::Clear() {
// @@protoc_insertion_point(message_clear_start:AN.ImageChunk)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.buffer_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ImageChunk::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // bytes buffer = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_buffer();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* ImageChunk::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:AN.ImageChunk)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // bytes buffer = 1;
  if (!this->_internal_buffer().empty()) {
    target = stream->WriteBytesMaybeAliased(
        1, this->_internal_buffer(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:AN.ImageChunk)
  return target;
}

size_t ImageChunk::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:AN.ImageChunk)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // bytes buffer = 1;
  if (!this->_internal_buffer().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_buffer());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ImageChunk::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ImageChunk::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ImageChunk::GetClassData() const { return &_class_data_; }


void ImageChunk::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ImageChunk*>(&to_msg);
  auto& from = static_cast<const ImageChunk&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:AN.ImageChunk)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_buffer().empty()) {
    _this->_internal_set_buffer(from._internal_buffer());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ImageChunk::CopyFrom(const ImageChunk& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:AN.ImageChunk)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ImageChunk::IsInitialized() const {
  return true;
}

void ImageChunk::InternalSwap(ImageChunk* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.buffer_, lhs_arena,
      &other->_impl_.buffer_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata ImageChunk::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_fridge_2eproto_getter, &descriptor_table_fridge_2eproto_once,
      file_level_metadata_fridge_2eproto[0]);
}

// ===================================================================

class DetectResult::_Internal {
 public:
};

DetectResult::DetectResult(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:AN.DetectResult)
}
DetectResult::DetectResult(const DetectResult& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  DetectResult* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.detection_){from._impl_.detection_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:AN.DetectResult)
}

inline void DetectResult::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.detection_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

DetectResult::~DetectResult() {
  // @@protoc_insertion_point(destructor:AN.DetectResult)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void DetectResult::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.detection_.~RepeatedPtrField();
}

void DetectResult::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void DetectResult::Clear() {
// @@protoc_insertion_point(message_clear_start:AN.DetectResult)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.detection_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DetectResult::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .AN.Detection detection = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_detection(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* DetectResult::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:AN.DetectResult)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .AN.Detection detection = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_detection_size()); i < n; i++) {
    const auto& repfield = this->_internal_detection(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:AN.DetectResult)
  return target;
}

size_t DetectResult::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:AN.DetectResult)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .AN.Detection detection = 1;
  total_size += 1UL * this->_internal_detection_size();
  for (const auto& msg : this->_impl_.detection_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData DetectResult::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    DetectResult::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*DetectResult::GetClassData() const { return &_class_data_; }


void DetectResult::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<DetectResult*>(&to_msg);
  auto& from = static_cast<const DetectResult&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:AN.DetectResult)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.detection_.MergeFrom(from._impl_.detection_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void DetectResult::CopyFrom(const DetectResult& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:AN.DetectResult)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DetectResult::IsInitialized() const {
  return true;
}

void DetectResult::InternalSwap(DetectResult* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.detection_.InternalSwap(&other->_impl_.detection_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DetectResult::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_fridge_2eproto_getter, &descriptor_table_fridge_2eproto_once,
      file_level_metadata_fridge_2eproto[1]);
}

// ===================================================================

class Detection::_Internal {
 public:
};

Detection::Detection(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:AN.Detection)
}
Detection::Detection(const Detection& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  Detection* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.x1_){}
    , decltype(_impl_.y1_){}
    , decltype(_impl_.x2_){}
    , decltype(_impl_.y2_){}
    , decltype(_impl_.confidence_){}
    , decltype(_impl_.id_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&_impl_.x1_, &from._impl_.x1_,
    static_cast<size_t>(reinterpret_cast<char*>(&_impl_.id_) -
    reinterpret_cast<char*>(&_impl_.x1_)) + sizeof(_impl_.id_));
  // @@protoc_insertion_point(copy_constructor:AN.Detection)
}

inline void Detection::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.x1_){0}
    , decltype(_impl_.y1_){0}
    , decltype(_impl_.x2_){0}
    , decltype(_impl_.y2_){0}
    , decltype(_impl_.confidence_){0}
    , decltype(_impl_.id_){0}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

Detection::~Detection() {
  // @@protoc_insertion_point(destructor:AN.Detection)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Detection::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void Detection::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void Detection::Clear() {
// @@protoc_insertion_point(message_clear_start:AN.Detection)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&_impl_.x1_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&_impl_.id_) -
      reinterpret_cast<char*>(&_impl_.x1_)) + sizeof(_impl_.id_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Detection::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // float x1 = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 13)) {
          _impl_.x1_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // float y1 = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 21)) {
          _impl_.y1_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // float x2 = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 29)) {
          _impl_.x2_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // float y2 = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 37)) {
          _impl_.y2_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // float confidence = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 45)) {
          _impl_.confidence_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // int32 id = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 48)) {
          _impl_.id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Detection::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:AN.Detection)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // float x1 = 1;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x1 = this->_internal_x1();
  uint32_t raw_x1;
  memcpy(&raw_x1, &tmp_x1, sizeof(tmp_x1));
  if (raw_x1 != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(1, this->_internal_x1(), target);
  }

  // float y1 = 2;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y1 = this->_internal_y1();
  uint32_t raw_y1;
  memcpy(&raw_y1, &tmp_y1, sizeof(tmp_y1));
  if (raw_y1 != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(2, this->_internal_y1(), target);
  }

  // float x2 = 3;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x2 = this->_internal_x2();
  uint32_t raw_x2;
  memcpy(&raw_x2, &tmp_x2, sizeof(tmp_x2));
  if (raw_x2 != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(3, this->_internal_x2(), target);
  }

  // float y2 = 4;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y2 = this->_internal_y2();
  uint32_t raw_y2;
  memcpy(&raw_y2, &tmp_y2, sizeof(tmp_y2));
  if (raw_y2 != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(4, this->_internal_y2(), target);
  }

  // float confidence = 5;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_confidence = this->_internal_confidence();
  uint32_t raw_confidence;
  memcpy(&raw_confidence, &tmp_confidence, sizeof(tmp_confidence));
  if (raw_confidence != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(5, this->_internal_confidence(), target);
  }

  // int32 id = 6;
  if (this->_internal_id() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(6, this->_internal_id(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:AN.Detection)
  return target;
}

size_t Detection::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:AN.Detection)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // float x1 = 1;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x1 = this->_internal_x1();
  uint32_t raw_x1;
  memcpy(&raw_x1, &tmp_x1, sizeof(tmp_x1));
  if (raw_x1 != 0) {
    total_size += 1 + 4;
  }

  // float y1 = 2;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y1 = this->_internal_y1();
  uint32_t raw_y1;
  memcpy(&raw_y1, &tmp_y1, sizeof(tmp_y1));
  if (raw_y1 != 0) {
    total_size += 1 + 4;
  }

  // float x2 = 3;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x2 = this->_internal_x2();
  uint32_t raw_x2;
  memcpy(&raw_x2, &tmp_x2, sizeof(tmp_x2));
  if (raw_x2 != 0) {
    total_size += 1 + 4;
  }

  // float y2 = 4;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y2 = this->_internal_y2();
  uint32_t raw_y2;
  memcpy(&raw_y2, &tmp_y2, sizeof(tmp_y2));
  if (raw_y2 != 0) {
    total_size += 1 + 4;
  }

  // float confidence = 5;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_confidence = this->_internal_confidence();
  uint32_t raw_confidence;
  memcpy(&raw_confidence, &tmp_confidence, sizeof(tmp_confidence));
  if (raw_confidence != 0) {
    total_size += 1 + 4;
  }

  // int32 id = 6;
  if (this->_internal_id() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_id());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Detection::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    Detection::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Detection::GetClassData() const { return &_class_data_; }


void Detection::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<Detection*>(&to_msg);
  auto& from = static_cast<const Detection&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:AN.Detection)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x1 = from._internal_x1();
  uint32_t raw_x1;
  memcpy(&raw_x1, &tmp_x1, sizeof(tmp_x1));
  if (raw_x1 != 0) {
    _this->_internal_set_x1(from._internal_x1());
  }
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y1 = from._internal_y1();
  uint32_t raw_y1;
  memcpy(&raw_y1, &tmp_y1, sizeof(tmp_y1));
  if (raw_y1 != 0) {
    _this->_internal_set_y1(from._internal_y1());
  }
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_x2 = from._internal_x2();
  uint32_t raw_x2;
  memcpy(&raw_x2, &tmp_x2, sizeof(tmp_x2));
  if (raw_x2 != 0) {
    _this->_internal_set_x2(from._internal_x2());
  }
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_y2 = from._internal_y2();
  uint32_t raw_y2;
  memcpy(&raw_y2, &tmp_y2, sizeof(tmp_y2));
  if (raw_y2 != 0) {
    _this->_internal_set_y2(from._internal_y2());
  }
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_confidence = from._internal_confidence();
  uint32_t raw_confidence;
  memcpy(&raw_confidence, &tmp_confidence, sizeof(tmp_confidence));
  if (raw_confidence != 0) {
    _this->_internal_set_confidence(from._internal_confidence());
  }
  if (from._internal_id() != 0) {
    _this->_internal_set_id(from._internal_id());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Detection::CopyFrom(const Detection& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:AN.Detection)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Detection::IsInitialized() const {
  return true;
}

void Detection::InternalSwap(Detection* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Detection, _impl_.id_)
      + sizeof(Detection::_impl_.id_)
      - PROTOBUF_FIELD_OFFSET(Detection, _impl_.x1_)>(
          reinterpret_cast<char*>(&_impl_.x1_),
          reinterpret_cast<char*>(&other->_impl_.x1_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Detection::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_fridge_2eproto_getter, &descriptor_table_fridge_2eproto_once,
      file_level_metadata_fridge_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace AN
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::AN::ImageChunk*
Arena::CreateMaybeMessage< ::AN::ImageChunk >(Arena* arena) {
  return Arena::CreateMessageInternal< ::AN::ImageChunk >(arena);
}
template<> PROTOBUF_NOINLINE ::AN::DetectResult*
Arena::CreateMaybeMessage< ::AN::DetectResult >(Arena* arena) {
  return Arena::CreateMessageInternal< ::AN::DetectResult >(arena);
}
template<> PROTOBUF_NOINLINE ::AN::Detection*
Arena::CreateMaybeMessage< ::AN::Detection >(Arena* arena) {
  return Arena::CreateMessageInternal< ::AN::Detection >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>