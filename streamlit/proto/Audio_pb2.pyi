"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
*!
Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Audio(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    URL_FIELD_NUMBER: builtins.int
    START_TIME_FIELD_NUMBER: builtins.int
    END_TIME_FIELD_NUMBER: builtins.int
    LOOP_FIELD_NUMBER: builtins.int
    url: builtins.str
    start_time: builtins.int
    """The currentTime attribute of the HTML <audio> tag's <source> subtag."""
    end_time: builtins.int
    """The time at which the audio should stop playing. If not specified, plays to the end."""
    loop: builtins.bool
    """Indicates whether the audio should start over from the beginning once it ends."""
    def __init__(
        self,
        *,
        url: builtins.str = ...,
        start_time: builtins.int = ...,
        end_time: builtins.int = ...,
        loop: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["end_time", b"end_time", "loop", b"loop", "start_time", b"start_time", "url", b"url"]) -> None: ...

global___Audio = Audio