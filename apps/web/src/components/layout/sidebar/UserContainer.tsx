import { Avatar } from "@heroui/avatar";
import { Button } from "@heroui/button";
import Image from "next/image";
import React from "react";
import { ChevronsDownUp, ChevronsUpDown } from "@/components/shared/icons";
import { useUser } from "@/features/auth/hooks/useUser";
import { useUserSubscriptionStatus } from "@/features/pricing/hooks/usePricing";
import SettingsMenu from "@/features/settings/components/SettingsMenu";

export default function UserContainer() {
  const user = useUser();
  const { data: subscriptionStatus } = useUserSubscriptionStatus();
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <SettingsMenu onOpenChange={setIsOpen}>
      <Button
        variant="light"
        className="group/triggerbtn pointer-events-auto relative flex w-full flex-row justify-between"
        endContent={
          isOpen ? (
            <ChevronsDownUp
              className="text-zinc-500 transition"
              width={20}
              height={20}
            />
          ) : (
            <ChevronsUpDown
              className="text-zinc-500 transition"
              width={20}
              height={20}
            />
          )
        }
      >
        <div className="flex items-center gap-3">
          <Avatar
            size="sm"
            radius="full"
            src={user?.profilePicture}
            alt="User Avatar"
            name="User"
            showFallback
            fallback={
              <Image
                src={"/images/avatars/default.webp"}
                width={30}
                height={30}
                alt="Default profile picture"
              />
            }
            classNames={{ base: "size-7" }}
          />
          <div className="flex flex-col items-start -space-y-0.5">
            <span className="text-sm">{user?.name}</span>
            {/* Render the plan label only once status resolves, so a paid user
                never briefly reads "GAIA Free" while the query is loading. */}
            {subscriptionStatus && (
              <span className="text-xs text-foreground-400">
                {subscriptionStatus.is_subscribed ? "GAIA Pro" : "GAIA Free"}
              </span>
            )}
          </div>
        </div>
      </Button>
    </SettingsMenu>
  );
}
