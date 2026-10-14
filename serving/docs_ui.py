from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse


def add_custom_docs(app):

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui():

        html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} | API Documentation",
            swagger_ui_parameters={
                "docExpansion": "list",
                "defaultModelsExpandDepth": -1,
                "displayRequestDuration": True,
            },
        )

        custom_html = html.body.decode("utf-8").replace(
            "</head>",
            """
            <style>

                body {
                    background: #F6F8FB;
                    font-family: "Segoe UI", Arial, sans-serif;
                }

                .swagger-ui .topbar {
                    display: none;
                }

                .swagger-ui .wrapper {
                    max-width: 1180px;
                }

                .swagger-ui .info {
                    margin: 36px 0 28px;
                }

                .swagger-ui .info .title {
                    color: #14233B;
                    font-size: 36px;
                    font-weight: 700;
                }

                .swagger-ui .info p,
                .swagger-ui .info li {
                    color: #667085;
                    line-height: 1.6;
                }

                .swagger-ui .opblock {
                    background: #FFFFFF !important;
                    border: 1px solid #DCE2EA !important;
                    border-radius: 12px;
                    box-shadow: none;
                    overflow: hidden;
                }

                .swagger-ui .opblock .opblock-summary-method {
                    background: #243B5A !important;
                    color: #FFFFFF !important;
                    border-radius: 6px;
                    font-weight: 600;
                }

                .swagger-ui .opblock-summary-path {
                    color: #14233B !important;
                    font-weight: 600;
                }

                .swagger-ui .opblock-summary-description {
                    color: #667085 !important;
                }

                .swagger-ui .btn.execute {
                    background: #243B5A !important;
                    border-color: #243B5A !important;
                    border-radius: 8px;
                }

                .swagger-ui .btn.try-out__btn {
                    color: #243B5A;
                    border-color: #98A2B3;
                    border-radius: 8px;
                }

                .swagger-ui input,
                .swagger-ui textarea,
                .swagger-ui select {
                    border: 1px solid #D0D5DD !important;
                    border-radius: 8px !important;
                    box-shadow: none !important;
                }

                .swagger-ui .model-box,
                .swagger-ui section.models {
                    background: #FFFFFF;
                    border: 1px solid #E4E7EC;
                    border-radius: 12px;
                }

                .swagger-ui pre {
                    background: #182230 !important;
                    border-radius: 8px;
                }

            </style>
            </head>
            """
        )

        return HTMLResponse(content=custom_html)