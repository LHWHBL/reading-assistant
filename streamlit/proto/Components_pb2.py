# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: streamlit/proto/Components.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n streamlit/proto/Components.proto\"\x8b\x01\n\x11\x43omponentInstance\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tjson_args\x18\x02 \x01(\t\x12!\n\x0cspecial_args\x18\x03 \x03(\x0b\x32\x0b.SpecialArg\x12\x16\n\x0e\x63omponent_name\x18\x04 \x01(\t\x12\x0b\n\x03url\x18\x05 \x01(\t\x12\x0f\n\x07\x66orm_id\x18\x06 \x01(\t\"_\n\nSpecialArg\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x0f\x61rrow_dataframe\x18\x02 \x01(\x0b\x32\x0f.ArrowDataframeH\x00\x12\x0f\n\x05\x62ytes\x18\x03 \x01(\x0cH\x00\x42\x07\n\x05value\"J\n\x0e\x41rrowDataframe\x12\x19\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x0b.ArrowTable\x12\x0e\n\x06height\x18\x02 \x01(\r\x12\r\n\x05width\x18\x03 \x01(\r\"]\n\nArrowTable\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05index\x18\x02 \x01(\x0c\x12\x0f\n\x07\x63olumns\x18\x03 \x01(\x0c\x12!\n\x06styler\x18\x05 \x01(\x0b\x32\x11.ArrowTableStyler\"Y\n\x10\x41rrowTableStyler\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x0f\n\x07\x63\x61ption\x18\x02 \x01(\t\x12\x0e\n\x06styles\x18\x03 \x01(\t\x12\x16\n\x0e\x64isplay_values\x18\x04 \x01(\x0c\x42/\n\x1c\x63om.snowflake.apps.streamlitB\x0f\x43omponentsProtob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'streamlit.proto.Components_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\034com.snowflake.apps.streamlitB\017ComponentsProto'
  _COMPONENTINSTANCE._serialized_start=37
  _COMPONENTINSTANCE._serialized_end=176
  _SPECIALARG._serialized_start=178
  _SPECIALARG._serialized_end=273
  _ARROWDATAFRAME._serialized_start=275
  _ARROWDATAFRAME._serialized_end=349
  _ARROWTABLE._serialized_start=351
  _ARROWTABLE._serialized_end=444
  _ARROWTABLESTYLER._serialized_start=446
  _ARROWTABLESTYLER._serialized_end=535
# @@protoc_insertion_point(module_scope)
