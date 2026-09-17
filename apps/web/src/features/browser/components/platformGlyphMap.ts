import type { BotPlatform } from "@/config/botPlatforms";
import {
  Discord,
  type Glyph,
  IMessage,
  Slack,
  Telegram,
  WhatsApp,
} from "./platformGlyphs";

export const PLATFORM_GLYPHS: Record<BotPlatform, Glyph> = {
  telegram: Telegram,
  whatsapp: WhatsApp,
  slack: Slack,
  discord: Discord,
  imessage: IMessage,
};
