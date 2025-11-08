# scanner/tracing_setup.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
import logging

def init_tracer(service_name="crypto-futures-scanner"):
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    try:
        jaeger_exporter = JaegerExporter(
            # arahkan langsung ke IP internal container
            agent_host_name="172.24.0.2",
            agent_port=6831,
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)

        logging.info("✅ Jaeger tracer initialized to 172.24.0.2:6831")
    except Exception as e:
        logging.warning(f"⚠️ Jaeger tracer disabled: {e}")

        # fallback supaya tidak error jika Jaeger mati
        class DummyTracer:
            def start_as_current_span(self, *a, **kw):
                from contextlib import nullcontext
                return nullcontext()
        return DummyTracer()

    return trace.get_tracer(service_name)
