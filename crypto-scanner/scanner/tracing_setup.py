# scanner/tracing_setup.py
"""
Tracing Setup v2 — Jaeger Integration Polished
- Sets proper service name via Resource
- Adds metadata tags for better trace visibility
- Gracefully degrades when Jaeger not reachable
"""

import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter


def init_tracer(service_name="crypto-scanner", agent_host="172.24.0.2", agent_port=6831):
    """
    Initialize OpenTelemetry tracer with Jaeger exporter.
    If Jaeger is not reachable, return a safe dummy tracer to avoid crashes.
    """

    # Proper resource naming (this fixes 'unknown_service')
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    try:
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=agent_host,
            agent_port=agent_port,
        )

        # Add processor to provider
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)

        logging.info(f"✅ Jaeger tracer initialized to {agent_host}:{agent_port} (service={service_name})")

        return trace.get_tracer(service_name)

    except Exception as e:
        logging.warning(f"⚠️ Jaeger tracer disabled: {e}")

        # Fallback no-op tracer (won’t break anything)
        class DummyTracer:
            def start_as_current_span(self, *a, **kw):
                from contextlib import nullcontext
                return nullcontext()
        return DummyTracer()
