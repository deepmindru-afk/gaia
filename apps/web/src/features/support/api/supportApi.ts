import type {
  SupportRequestCreate,
  SupportRequestSubmissionResponse,
} from "@shared/api/generated";
import { apiauth } from "@/lib/api/client";

export type {
  SupportRequestCreate,
  SupportRequestSubmissionResponse,
} from "@shared/api/generated";

class SupportApiService {
  /**
   * Submit a support or feature request
   */
  async submitRequest(
    requestData: SupportRequestCreate,
    attachments?: File[],
  ): Promise<SupportRequestSubmissionResponse> {
    try {
      // If there are attachments, use FormData
      if (attachments && attachments.length > 0) {
        const formData = new FormData();
        formData.append("type", requestData.type);
        formData.append("title", requestData.title);
        formData.append("description", requestData.description);

        // Append each attachment
        attachments.forEach((file) => {
          formData.append("attachments", file);
        });

        const response = await apiauth.post<SupportRequestSubmissionResponse>(
          "support/requests/with-attachments",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          },
        );
        return response.data;
      } else {
        // No attachments, use regular JSON
        const response = await apiauth.post<SupportRequestSubmissionResponse>(
          "support/requests",
          requestData,
        );
        return response.data;
      }
    } catch (error) {
      console.error("Error submitting support request:", error);
      throw new Error("Failed to submit support request");
    }
  }
}

export const supportApi = new SupportApiService();
