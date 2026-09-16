import { api, binaryField, formDataSerializer } from "@/lib/api/typed";

export interface SupportRequest {
  type: "support" | "feature";
  title: string;
  description: string;
  attachments?: File[];
}

export const supportApi = {
  /**
   * Submit a support or feature request.
   *
   * Attachments go to the multipart route; the body is the schema's own type
   * either way, so a renamed field fails to compile instead of at runtime.
   */
  submitRequest: ({ type, title, description, attachments }: SupportRequest) =>
    attachments && attachments.length > 0
      ? api.post("/api/v1/support/requests/with-attachments", {
          body: {
            type,
            title,
            description,
            attachments: attachments.map(binaryField),
          },
          bodySerializer: formDataSerializer,
          errorMessage: "Failed to submit support request",
        })
      : api.post("/api/v1/support/requests", {
          body: { type, title, description },
          errorMessage: "Failed to submit support request",
        }),
};
