import { apiauth } from "@/lib/api/client";

/**
 * KNOWN BROKEN: the API serves no `/api/v1/team` route — it is absent from
 * `apps/api/openapi.json`, so every call here 404s and the blog editor's
 * author picker has never been able to populate. That is also why this is the
 * one caller left on the raw axios instance: the path-typed client only
 * accepts paths the schema declares, and there is nothing to type against.
 * Either the API grows the route or this module and the author picker in
 * `CreateBlogPage` go — it is a product call, not a mechanical one.
 */

export interface TeamMember {
  id: string;
  name: string;
  role: string;
  avatar?: string;
  linkedin?: string;
  twitter?: string;
}

export const teamApi = {
  getTeamMembers: async (): Promise<TeamMember[]> => {
    const response = await apiauth.get<TeamMember[]>("/team");
    return response.data;
  },
};
