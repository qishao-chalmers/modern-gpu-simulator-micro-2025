# Protobuf (and Abseil) link flags for Makefile builds.
# Protobuf >= 4.x (e.g. EasyBuild 24.x) depends on Abseil DSOs that must appear on the link line.

ifndef PROTOBUF_DEPS_MK_INCLUDED
PROTOBUF_DEPS_MK_INCLUDED := 1

PROTOBUF_PREFIX := $(shell pkg-config --variable=prefix protobuf 2>/dev/null)
PROTOBUF_INCLUDEDIR := $(shell pkg-config --variable=includedir protobuf 2>/dev/null)
PROTOBUF_LIBDIR := $(shell pkg-config --variable=libdir protobuf 2>/dev/null)

ifeq ($(PROTOBUF_INCLUDEDIR),)
  ifneq ($(PROTOBUF_PREFIX),)
    PROTOBUF_INCLUDEDIR := $(PROTOBUF_PREFIX)/include
  endif
endif
ifeq ($(PROTOBUF_LIBDIR),)
  ifneq ($(PROTOBUF_PREFIX),)
    PROTOBUF_LIBDIR := $(PROTOBUF_PREFIX)/lib
  endif
endif

# pkg-config --cflags on some EasyBuild installs omits -I; prepend non-default include dirs.
_PROTOBUF_CFLAGS_PC := $(shell pkg-config --cflags protobuf 2>/dev/null)
PROTOBUF_CFLAGS := $(_PROTOBUF_CFLAGS_PC)
ifneq ($(PROTOBUF_INCLUDEDIR),)
  ifeq ($(findstring -I$(PROTOBUF_INCLUDEDIR),$(PROTOBUF_CFLAGS)),)
    ifneq ($(PROTOBUF_INCLUDEDIR),/usr/include)
      # Do not force -I/usr/include: it breaks g++ system header search (#include_next).
      PROTOBUF_CFLAGS := -I$(PROTOBUF_INCLUDEDIR) $(PROTOBUF_CFLAGS)
    endif
  endif
endif

PROTOBUF_LIBS := $(shell pkg-config --libs protobuf 2>/dev/null)
ifneq ($(PROTOBUF_LIBDIR),)
  ifeq ($(findstring -L$(PROTOBUF_LIBDIR),$(PROTOBUF_LIBS)),)
    PROTOBUF_LIBS := -L$(PROTOBUF_LIBDIR) $(PROTOBUF_LIBS)
  endif
endif

# Use protoc from the same prefix as pkg-config protobuf (critical on module systems).
ifndef PROTOC
  ifneq ($(PROTOBUF_PREFIX),)
    PROTOC := $(PROTOBUF_PREFIX)/bin/protoc
  else
    PROTOC := protoc
  endif
endif

ifeq ($(PROTOBUF_LIBS),)
  PROTOBUF_CFLAGS ?=
  PROTOBUF_LIBS := -lprotobuf
else
  # pkg-config --modversion returns e.g. "24.0" or "3.12.4"; first field is the major.
  PROTOBUF_NEEDS_ABSEIL := $(shell \
    v=$$(pkg-config --modversion protobuf 2>/dev/null | cut -d. -f1); \
    if [ -n "$$v" ] && [ "$$v" -ge 4 ] 2>/dev/null; then echo 1; else echo 0; fi)
  ifeq ($(PROTOBUF_NEEDS_ABSEIL),1)
    PROTOBUF_STATIC_LIBS := $(shell pkg-config --static --libs protobuf 2>/dev/null)
    ifneq ($(PROTOBUF_STATIC_LIBS),)
      ifneq ($(PROTOBUF_STATIC_LIBS),$(PROTOBUF_LIBS))
        PROTOBUF_LIBS := $(PROTOBUF_STATIC_LIBS)
      endif
    endif

    # If static pkg-config still only returns -lprotobuf, append Abseil explicitly.
    ifeq ($(findstring absl,$(PROTOBUF_LIBS)),)
      ABSL_LIBS := $(shell pkg-config --libs \
        absl_log_internal_check_op \
        absl_log_internal_message \
        absl_log_internal_nullguard \
        absl_log_internal_format \
        absl_log_internal_globals \
        absl_log_internal_proto \
        absl_log_globals \
        absl_log_severity \
        absl_raw_logging_internal \
        absl_spinlock_wait \
        absl_strerror \
        absl_strings \
        absl_string_view \
        absl_base \
        absl_throw_delegate \
        absl_int128 \
        2>/dev/null)
      ifeq ($(ABSL_LIBS),)
        # Fallback when absl .pc files are unavailable (common on module systems).
        ABSL_LIBS := \
          -labsl_log_internal_check_op \
          -labsl_log_internal_message \
          -labsl_log_internal_nullguard \
          -labsl_log_internal_format \
          -labsl_log_internal_globals \
          -labsl_log_internal_proto \
          -labsl_log_globals \
          -labsl_log_severity \
          -labsl_raw_logging_internal \
          -labsl_spinlock_wait \
          -labsl_strerror \
          -labsl_strings \
          -labsl_string_view \
          -labsl_base \
          -labsl_throw_delegate \
          -labsl_int128
      endif
      PROTOBUF_LIBS += $(ABSL_LIBS)
    endif
  endif
endif

# Allow callers to append more libs after including this file.
PROTOBUF_LIBS +=

# Compiler-only flags (safe for g++). NVCC should only get PROTOBUF_INCLUDES (-I/-D).
PROTOBUF_INCLUDES := $(filter -I% -D%,$(PROTOBUF_CFLAGS))
PROTOBUF_COMPILE_FLAGS := $(filter-out -I% -D%,$(PROTOBUF_CFLAGS))

endif # PROTOBUF_DEPS_MK_INCLUDED
