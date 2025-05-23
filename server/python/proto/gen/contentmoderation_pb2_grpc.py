# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import contentmoderation_pb2 as contentmoderation__pb2

GRPC_GENERATED_VERSION = '1.71.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in contentmoderation_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ContentModerationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ModerateContent = channel.unary_unary(
                '/contentmoderation.ContentModerationService/ModerateContent',
                request_serializer=contentmoderation__pb2.ModerationRequest.SerializeToString,
                response_deserializer=contentmoderation__pb2.ModerationResponse.FromString,
                _registered_method=True)
        self.ModerateContentAsync = channel.unary_unary(
                '/contentmoderation.ContentModerationService/ModerateContentAsync',
                request_serializer=contentmoderation__pb2.ModerationRequest.SerializeToString,
                response_deserializer=contentmoderation__pb2.AsyncModerationResponse.FromString,
                _registered_method=True)
        self.GetModerationResult = channel.unary_unary(
                '/contentmoderation.ContentModerationService/GetModerationResult',
                request_serializer=contentmoderation__pb2.ResultRequest.SerializeToString,
                response_deserializer=contentmoderation__pb2.ModerationResponse.FromString,
                _registered_method=True)
        self.ModerateBatchContent = channel.unary_unary(
                '/contentmoderation.ContentModerationService/ModerateBatchContent',
                request_serializer=contentmoderation__pb2.BatchModerationRequest.SerializeToString,
                response_deserializer=contentmoderation__pb2.BatchModerationResponse.FromString,
                _registered_method=True)


class ContentModerationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ModerateContent(self, request, context):
        """同步审核接口
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModerateContentAsync(self, request, context):
        """异步审核接口
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModerationResult(self, request, context):
        """获取异步审核结果
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModerateBatchContent(self, request, context):
        """批量审核接口
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ContentModerationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ModerateContent': grpc.unary_unary_rpc_method_handler(
                    servicer.ModerateContent,
                    request_deserializer=contentmoderation__pb2.ModerationRequest.FromString,
                    response_serializer=contentmoderation__pb2.ModerationResponse.SerializeToString,
            ),
            'ModerateContentAsync': grpc.unary_unary_rpc_method_handler(
                    servicer.ModerateContentAsync,
                    request_deserializer=contentmoderation__pb2.ModerationRequest.FromString,
                    response_serializer=contentmoderation__pb2.AsyncModerationResponse.SerializeToString,
            ),
            'GetModerationResult': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModerationResult,
                    request_deserializer=contentmoderation__pb2.ResultRequest.FromString,
                    response_serializer=contentmoderation__pb2.ModerationResponse.SerializeToString,
            ),
            'ModerateBatchContent': grpc.unary_unary_rpc_method_handler(
                    servicer.ModerateBatchContent,
                    request_deserializer=contentmoderation__pb2.BatchModerationRequest.FromString,
                    response_serializer=contentmoderation__pb2.BatchModerationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'contentmoderation.ContentModerationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('contentmoderation.ContentModerationService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ContentModerationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ModerateContent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/contentmoderation.ContentModerationService/ModerateContent',
            contentmoderation__pb2.ModerationRequest.SerializeToString,
            contentmoderation__pb2.ModerationResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ModerateContentAsync(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/contentmoderation.ContentModerationService/ModerateContentAsync',
            contentmoderation__pb2.ModerationRequest.SerializeToString,
            contentmoderation__pb2.AsyncModerationResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetModerationResult(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/contentmoderation.ContentModerationService/GetModerationResult',
            contentmoderation__pb2.ResultRequest.SerializeToString,
            contentmoderation__pb2.ModerationResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ModerateBatchContent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/contentmoderation.ContentModerationService/ModerateBatchContent',
            contentmoderation__pb2.BatchModerationRequest.SerializeToString,
            contentmoderation__pb2.BatchModerationResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
